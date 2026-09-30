"""Stage `train`: distil the teacher's soft labels into the student. Early-stops on validation
agreement with the teacher; the full run log lives in runs/<run_id>/."""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import Decision, decisions
from .losses import IGNORE, total_loss
from .model import build_model, count_params, save_checkpoint
from .stages import Ctx
from .tok import PAD_ID, load_tokenizer
from .util import log, pick_device, read_jsonl, sha256_file, write_json


class Encoded:
    """Pre-tokenised rows: ragged id lists + teacher distributions + weak hard labels."""

    def __init__(self, rows: list[dict[str, Any]], tokenizer_path: Path, decs: list[Decision], max_len: int) -> None:
        tok = load_tokenizer(tokenizer_path)
        tok.no_padding()
        tok.enable_truncation(max_length=max_len)
        self.ids = [e.ids for e in tok.encode_batch([r["text"] for r in rows])]
        self.names = [d.name for d in decs]
        self.probs = {d.name: torch.tensor([r["probs"][d.name] for r in rows], dtype=torch.float32) for d in decs}
        self.hard = {}
        for d in decs:
            idx = {lab: i for i, lab in enumerate(d.labels)}
            self.hard[d.name] = torch.tensor([idx.get(str((r.get("weak") or {}).get(d.name)), IGNORE) for r in rows],
                                             dtype=torch.long)

    def __len__(self) -> int:
        return len(self.ids)

    def batch(self, idx: list[int] | np.ndarray, device: str) -> tuple[Tensor, Tensor, dict[str, Tensor], dict[str, Tensor]]:
        seqs = [self.ids[i] for i in idx]
        t = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), t), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(seqs), t), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        sel = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        probs = {k: v[sel].to(device) for k, v in self.probs.items()}
        hard = {k: v[sel].to(device) for k, v in self.hard.items()}
        return ids.to(device), mask.to(device), probs, hard


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict_logits(model: torch.nn.Module, data: Encoded, device: str, batch_size: int = 256) -> dict[str, Tensor]:
    model.eval()
    out: dict[str, list[Tensor]] = {n: [] for n in data.names}
    order = np.arange(len(data))
    for i in range(0, len(order), batch_size):
        ids, mask, _, _ = data.batch(order[i:i + batch_size], device)
        for n, z in zip(data.names, model(ids, mask)):
            out[n].append(z.float().cpu())
    return {n: torch.cat(v) for n, v in out.items()}


def evaluate_val(logits: dict[str, Tensor], data: Encoded, temperature: float) -> dict[str, float]:
    from .losses import kd_loss

    m: dict[str, float] = {}
    for n, z in logits.items():
        m[f"agree/{n}"] = float((z.argmax(-1) == data.probs[n].argmax(-1)).float().mean())
        m[f"kd/{n}"] = float(kd_loss(z, data.probs[n], temperature))
        h = data.hard[n]
        keep = h != IGNORE
        if bool(keep.any()):
            m[f"weak_acc/{n}"] = float((z.argmax(-1)[keep] == h[keep]).float().mean())
    m["agree/mean"] = float(np.mean([m[f"agree/{n}"] for n in logits]))
    return m


def run(ctx: Ctx) -> dict[str, Any]:
    cfg, tcfg = ctx.cfg, ctx.cfg["train"]
    decs = decisions(cfg)
    max_len = int(cfg["model"]["encoder"]["max_len"])
    tok_path = ctx.path("tokenizer", "tokenizer.json")
    device = pick_device(tcfg.get("device", "auto"))
    train = Encoded(read_jsonl(ctx.path("split", "train.jsonl")), tok_path, decs, max_len)
    val = Encoded(read_jsonl(ctx.path("split", "val.jsonl")), tok_path, decs, max_len)
    vocab = load_tokenizer(tok_path).get_vocab_size()
    heads_out = {d.name: d.n for d in decs}
    torch.manual_seed(int(cfg["seed"]))
    model = build_model(cfg["model"], vocab, heads_out).to(device)
    n_params = count_params(model)

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{cfg['profile']}"
    run_dir = ctx.root / cfg["runs_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg)
    log.info("run %s: %d train / %d val rows, %.2fM params, device %s", run_id, len(train), len(val), n_params / 1e6, device)

    bs, epochs = int(tcfg["batch_size"]), int(tcfg["epochs"])
    steps_per_epoch = math.ceil(len(train) / bs)
    total_steps = steps_per_epoch * epochs
    warmup = int(float(tcfg["warmup_ratio"]) * total_steps)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "ln" in name or name.startswith("pos") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(tcfg["weight_decay"])},
                             {"params": no_decay, "weight_decay": 0.0}], lr=float(tcfg["lr"]))
    T, hw = float(tcfg["kd_temperature"]), float(tcfg["hard_weight"])
    head_w = {k: float(v) for k, v in (tcfg.get("head_weights") or {}).items()}
    rng = np.random.default_rng(int(cfg["seed"]))
    best, best_epoch, bad, step = -1.0, -1, 0, 0
    best_path = run_dir / "model.pt"
    meta = {"vocab_size": vocab, "heads_out": heads_out, "labels": {d.name: list(d.labels) for d in decs},
            "max_len": max_len, "tokenizer_sha256": sha256_file(tok_path), "params": n_params, "run_id": run_id,
            "task_version": cfg["task"].get("version")}
    metrics_f = open(run_dir / "metrics.jsonl", "w", encoding="utf-8")
    t_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        t0, run_loss = time.perf_counter(), 0.0
        for i in range(0, len(order), bs):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, warmup, float(tcfg["lr"]))
            ids, mask, probs, hard = train.batch(order[i:i + bs], device)
            logits = dict(zip(train.names, model(ids, mask)))
            loss, _ = total_loss(logits, probs, hard, head_w, T, hw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += float(loss.detach())
            step += 1
        vm = evaluate_val(predict_logits(model, val, device), val, T)
        rec = {"epoch": epoch + 1, "step": step, "train_loss": run_loss / steps_per_epoch,
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.perf_counter() - t0, 1), **vm}
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        log.info("epoch %2d loss %.4f  val agree %s  (%.0fs)", epoch + 1, rec["train_loss"],
                 " ".join(f"{d.name}={vm[f'agree/{d.name}']:.3f}" for d in decs), rec["seconds"])
        # Early-stop score: teacher agreement, with weak-label accuracy as a tie-breaker.
        score = vm["agree/mean"] + 1e-3 * float(np.mean([v for k, v in vm.items() if k.startswith("weak_acc/")] or [0]))
        if score > best:
            best, best_epoch, bad = score, epoch + 1, 0
            save_checkpoint(best_path, model, {**meta, "epoch": epoch + 1, "val": vm})
        else:
            bad += 1
            if bad >= int(tcfg["patience"]):
                log.info("early stop at epoch %d (best %d)", epoch + 1, best_epoch)
                break
    metrics_f.close()
    out = ctx.path("train", "model.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, out)
    summary = {"run_id": run_id, "best_epoch": best_epoch, "best_score": round(best, 4), "params": n_params,
               "minutes": round((time.perf_counter() - t_start) / 60, 2)}
    write_json(ctx.path("train", "train.json"), summary)
    return summary
