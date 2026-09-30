"""Stage `calibrate`: fit one temperature per head on the validation split.

Validation rows carry only teacher distributions, so the objective is the soft-target NLL
(cross-entropy against the teacher): the student inherits the teacher's calibration, and ECE is
reported against the teacher's argmax here and against gold labels in `eval`."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import decisions
from .metrics import ece
from .model import load_checkpoint
from .stages import Ctx
from .train import Encoded, predict_logits
from .util import log, read_jsonl, write_json


def soft_nll(logits: np.ndarray, target: np.ndarray, t: float) -> float:
    z = logits / t
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return float(-(target * logp).sum(-1).mean())


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(logits: np.ndarray, target: np.ndarray, lo: float, hi: float, points: int) -> float:
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), int(points)))
    losses = [soft_nll(logits, target, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def run(ctx: Ctx) -> dict[str, Any]:
    cfg = ctx.cfg
    decs = decisions(cfg)
    model, meta = load_checkpoint(ctx.path("train", "model.pt"), cfg["model"])
    val = Encoded(read_jsonl(ctx.path("split", "val.jsonl")), ctx.path("tokenizer", "tokenizer.json"), decs,
                  int(meta["max_len"]))
    with torch.no_grad():
        logits = {k: v.numpy() for k, v in predict_logits(model, val, "cpu").items()}
    lo, hi, pts = cfg["calibrate"]["grid"]
    out: dict[str, Any] = {"temperatures": {}, "val_rows": len(val), "heads": {}}
    for d in decs:
        z, p = logits[d.name], val.probs[d.name].numpy()
        t = fit_temperature(z, p, float(lo), float(hi), int(pts))
        y = p.argmax(-1)
        head = {"temperature": round(t, 4), "nll_before": soft_nll(z, p, 1.0), "nll_after": soft_nll(z, p, t),
                "ece_vs_teacher_before": ece(softmax(z), y), "ece_vs_teacher_after": ece(softmax(z / t), y)}
        out["temperatures"][d.name] = head["temperature"]
        out["heads"][d.name] = head
        log.info("calibrate %-10s T=%.3f  nll %.4f->%.4f  ece(teacher) %.4f->%.4f", d.name, t, head["nll_before"],
                 head["nll_after"], head["ece_vs_teacher_before"], head["ece_vs_teacher_after"])
    write_json(ctx.path("calibrate", "calibration.json"), out)
    return {"temperatures": out["temperatures"]}
