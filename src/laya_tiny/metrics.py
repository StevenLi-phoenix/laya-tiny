"""Metrics over probability arrays: accuracy, macro-F1, ECE, teacher agreement, TV distance."""
from __future__ import annotations

import numpy as np


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(-1) == y).mean()) if len(y) else float("nan")


def macro_f1(probs: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    pred = probs.argmax(-1)
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fp + fn == 0:
            continue                      # class absent from both: undefined, skip
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-1 confidence, equal-width bins."""
    if not len(y):
        return float("nan")
    conf = probs.max(-1)
    correct = (probs.argmax(-1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            total += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(total)


def agreement(probs: np.ndarray, teacher: np.ndarray) -> float:
    return float((probs.argmax(-1) == teacher.argmax(-1)).mean())


def tv_distance(probs: np.ndarray, teacher: np.ndarray) -> float:
    """Mean total-variation distance between two sets of distributions (0 = identical, 1 = disjoint)."""
    return float(0.5 * np.abs(probs - teacher).sum(-1).mean())


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else float("nan")
