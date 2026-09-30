"""The student: a small pre-LN Transformer encoder with one classification head per decision.

Attention is written out with plain MatMul/Softmax (no fused SDPA) so the graph exports to any
ONNX opset and dynamic int8 quantisation finds every weight matrix."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, bias: Tensor) -> Tensor:
        b, t, d = x.shape
        qkv = self.qkv(self.ln1(x)).reshape(b, t, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dh)) + bias
        att = self.drop(torch.softmax(att, dim=-1))
        out = torch.matmul(att, v).transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.proj(out))
        return x + self.drop(self.ff(self.ln2(x)))


class TinyEncoder(nn.Module):
    def __init__(self, vocab_size: int, heads_out: dict[str, int], layers: int = 4, d_model: int = 192,
                 heads: int = 4, ffn_mult: int = 4, max_len: int = 192, dropout: float = 0.1,
                 pooling: str = "mean") -> None:
        super().__init__()
        self.max_len, self.pooling = max_len, pooling
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(d_model, heads, ffn_mult * d_model, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(d_model)
        self.names = list(heads_out)
        self.heads = nn.ModuleDict({k: nn.Linear(d_model, n) for k, n in heads_out.items()})
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, ...]:
        t = input_ids.shape[1]
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok(input_ids) + self.pos(pos))
        mask = attention_mask.to(x.dtype)
        bias = ((1.0 - mask) * -1e4)[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, bias)
        x = self.ln(x)
        if self.pooling == "cls":
            pooled = x[:, 0]
        else:
            m = mask.unsqueeze(-1)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return tuple(self.heads[k](pooled) for k in self.names)


def build_model(model_cfg: dict[str, Any], vocab_size: int, heads_out: dict[str, int]) -> TinyEncoder:
    e = model_cfg["encoder"]
    return TinyEncoder(vocab_size, heads_out, layers=int(e["layers"]), d_model=int(e["d_model"]),
                       heads=int(e["heads"]), ffn_mult=int(e["ffn_mult"]), max_len=int(e["max_len"]),
                       dropout=float(e.get("dropout", 0.1)), pooling=e.get("pooling", "mean"))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Calibrated(nn.Module):
    """Export wrapper: logits / T -> probabilities, one output per decision."""

    def __init__(self, model: TinyEncoder, temperatures: list[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("inv_t", torch.tensor([1.0 / t for t in temperatures]))

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, ...]:
        logits = self.model(input_ids, attention_mask)
        return tuple(torch.softmax(z * self.inv_t[i], dim=-1) for i, z in enumerate(logits))


def save_checkpoint(path: Any, model: TinyEncoder, meta: dict[str, Any]) -> None:
    state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "meta": meta}, path)


def load_checkpoint(path: Any, model_cfg: dict[str, Any]) -> tuple[TinyEncoder, dict[str, Any]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = build_model(model_cfg, meta["vocab_size"], meta["heads_out"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, meta
