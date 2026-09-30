"""Stage `export`: calibrated student -> ONNX (fp32), verified against PyTorch, then dynamic int8.

Outputs are probabilities (temperature already applied), one tensor per decision, named after it."""
from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import decisions
from .model import Calibrated, load_checkpoint
from .stages import Ctx
from .tok import load_tokenizer
from .util import iter_jsonl, log, sha256_file, write_json


def encode(tokenizer_path: Path, texts: list[str], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    tok = load_tokenizer(tokenizer_path)
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    encs = tok.encode_batch(texts)
    return (np.asarray([e.ids for e in encs], dtype=np.int64),
            np.asarray([e.attention_mask for e in encs], dtype=np.int64))


def export_onnx(wrapper: torch.nn.Module, names: list[str], path: Path, opset: int, max_len: int) -> None:
    ids = torch.randint(4, 100, (2, min(16, max_len)), dtype=torch.long)
    mask = torch.ones_like(ids)
    dyn = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}}
    dyn.update({n: {0: "batch"} for n in names})
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapper, (ids, mask), str(path), input_names=["input_ids", "attention_mask"],
                          output_names=names, dynamic_axes=dyn, opset_version=opset, do_constant_folding=True,
                          dynamo=False)


def ort_session(path: Path, threads: int = 1) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def quantize_int8(src: Path, dst: Path) -> None:
    import logging

    from onnxruntime.quantization import QuantType, quantize_dynamic

    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.WARNING)            # the quantiser logs every tensor at INFO on the root logger
    try:
        # Gather too: the token-embedding table is ~half of all weights.
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=False,
                         op_types_to_quantize=["MatMul", "Gemm", "Gather"])
    finally:
        root.setLevel(level)


def compare(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, float]:
    diff = max(float(np.abs(x - y).max()) for x, y in zip(a, b))
    agree = float(np.mean([np.mean(x.argmax(-1) == y.argmax(-1)) for x, y in zip(a, b)]))
    return {"max_abs_diff": diff, "argmax_agreement": agree}


def run(ctx: Ctx) -> dict[str, Any]:
    cfg = ctx.cfg
    names = [d.name for d in decisions(cfg)]
    model, meta = load_checkpoint(ctx.path("train", "model.pt"), cfg["model"])
    temps = json.loads(ctx.path("calibrate", "calibration.json").read_text())["temperatures"]
    wrapper = Calibrated(model, [float(temps[n]) for n in names]).eval()
    out_dir = ctx.path("export")
    fp32, int8 = out_dir / "student.onnx", out_dir / "student.int8.onnx"
    max_len = int(meta["max_len"])
    export_onnx(wrapper, names, fp32, int(cfg["export"]["opset"]), max_len)
    quantize_int8(fp32, int8)
    tok_src = ctx.path("tokenizer", "tokenizer.json")
    shutil.copy2(tok_src, out_dir / "tokenizer.json")

    texts = [r["text"] for _, r in zip(range(128), iter_jsonl(ctx.path("split", "val.jsonl")))]
    ids, mask = encode(tok_src, texts, max_len)
    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(torch.from_numpy(ids), torch.from_numpy(mask))]
    feeds = {"input_ids": ids, "attention_mask": mask}
    got32 = ort_session(fp32).run(names, feeds)
    got8 = ort_session(int8).run(names, feeds)
    eq32, eq8 = compare(ref, got32), compare(got32, got8)
    atol = float(cfg["export"]["atol"])
    log.info("onnx fp32 vs torch: %s", eq32)
    log.info("onnx int8 vs fp32:  %s", eq8)
    if eq32["max_abs_diff"] > atol:
        raise AssertionError(f"ONNX fp32 deviates from PyTorch by {eq32['max_abs_diff']:.2e} > {atol:.0e}")
    labels_meta = json.loads(ctx.path("label", "meta.json").read_text()) if ctx.path("label", "meta.json").exists() else {}
    info = {
        "task": cfg["task"]["name"], "task_version": cfg["task"].get("version"), "outputs": names,
        "labels": meta["labels"], "max_len": max_len, "temperatures": temps, "params": meta["params"],
        "run_id": meta["run_id"], "teacher": labels_meta.get("teacher"),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                  for p in (fp32, int8, out_dir / "tokenizer.json")},
        "equivalence": {"fp32_vs_torch": eq32, "int8_vs_fp32": eq8, "rows": len(texts)},
        "inputs": {"input_ids": "int64 [batch, seq]", "attention_mask": "int64 [batch, seq]"},
    }
    write_json(out_dir / "meta.json", info)
    return {"fp32_bytes": fp32.stat().st_size, "int8_bytes": int8.stat().st_size,
            "fp32_max_abs_diff": eq32["max_abs_diff"], "int8_argmax_agreement": eq8["argmax_agreement"]}
