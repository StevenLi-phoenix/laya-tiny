"""Distillation objective: temperature-scaled KL to the teacher plus optional weak-label CE."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

IGNORE = -100


def kd_loss(student_logits: Tensor, teacher_probs: Tensor, temperature: float) -> Tensor:
    """KL(teacher_T || student_T) * T^2, where teacher_T = softmax(log p / T).

    The teacher gives calibrated probabilities, not logits, so log p stands in for its logits."""
    t = float(temperature)
    target = torch.softmax(torch.log(teacher_probs.clamp_min(1e-8)) / t, dim=-1)
    log_student = F.log_softmax(student_logits / t, dim=-1)
    return F.kl_div(log_student, target, reduction="batchmean") * (t * t)


def hard_loss(student_logits: Tensor, labels: Tensor) -> Tensor:
    if bool((labels != IGNORE).any()):
        return F.cross_entropy(student_logits, labels, ignore_index=IGNORE)
    return student_logits.sum() * 0.0


def total_loss(logits: dict[str, Tensor], teacher: dict[str, Tensor], hard: dict[str, Tensor],
               head_weights: dict[str, float], temperature: float, hard_weight: float) -> tuple[Tensor, dict[str, float]]:
    parts: dict[str, float] = {}
    total = next(iter(logits.values())).sum() * 0.0
    for name, z in logits.items():
        kd = kd_loss(z, teacher[name], temperature)
        loss = kd
        parts[f"kd/{name}"] = float(kd.detach())
        if hard_weight > 0 and name in hard:
            ce = hard_loss(z, hard[name])
            loss = loss + hard_weight * ce
            parts[f"ce/{name}"] = float(ce.detach())
        total = total + float(head_weights.get(name, 1.0)) * loss
    return total, parts
