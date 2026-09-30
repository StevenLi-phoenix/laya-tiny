"""Stage `split`: join corpus + teacher labels, drop anything near-duplicating the holdout
(leakage guard), and split train/val deterministically by content hash."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .data import stable_bucket
from .stages import Ctx
from .teacher import read_labels
from .util import log, read_jsonl, write_jsonl

_WORD = re.compile(r"[a-z0-9']+")


def shingles(text: str, n: int = 3) -> frozenset[str]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


class LeakIndex:
    """Inverted shingle index over the holdout; `max_jaccard` finds the closest holdout row."""

    def __init__(self, texts: list[str]) -> None:
        self.sets = [shingles(t) for t in texts]
        self.index: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(self.sets):
            for sh in s:
                self.index[sh].append(i)

    def max_jaccard(self, text: str) -> tuple[float, int]:
        s = shingles(text)
        if not s:
            return 0.0, -1
        hits: dict[int, int] = defaultdict(int)
        for sh in s:
            for i in self.index.get(sh, ()):
                hits[i] += 1
        best, arg = 0.0, -1
        for i, inter in hits.items():
            j = inter / (len(s) + len(self.sets[i]) - inter)
            if j > best:
                best, arg = j, i
        return best, arg


def run(ctx: Ctx) -> dict[str, Any]:
    dcfg = ctx.cfg["data"]
    corpus = read_jsonl(ctx.path("data", "corpus.jsonl"))
    labels = read_labels(ctx.path("label", "corpus.jsonl"))
    holdout = read_jsonl(ctx.holdout)
    leak = LeakIndex([h["text"] for h in holdout])
    thr = float(dcfg["leak_jaccard"])
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    dropped: list[tuple[float, str, str]] = []
    unlabelled = 0
    for r in corpus:
        if r["id"] not in labels:
            unlabelled += 1
            continue
        j, i = leak.max_jaccard(r["text"])
        if j >= thr:
            dropped.append((j, r["text"], holdout[i]["text"]))
            continue
        row = {**r, "probs": labels[r["id"]]}
        (val if stable_bucket(r["id"]) < float(dcfg["val_fraction"]) else train).append(row)
    for j, a, b in sorted(dropped, reverse=True)[:8]:
        log.info("leak %.2f  corpus=%r  holdout=%r", j, a[:70], b[:70])
    if unlabelled:
        raise RuntimeError(f"{unlabelled} corpus rows have no teacher label; re-run `label`")
    write_jsonl(ctx.path("split", "train.jsonl"), train)
    write_jsonl(ctx.path("split", "val.jsonl"), val)
    return {"train": len(train), "val": len(val), "leak_dropped": len(dropped), "leak_threshold": thr}
