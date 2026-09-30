"""Load and validate configs/{task,model,pipeline}.yaml into one dict, with optional profile overrides."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .util import log

DECISION_TYPES = {"choice", "score", "noul"}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Dicts merge recursively; anything else (lists included) is replaced."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass(frozen=True)
class Decision:
    name: str
    type: str
    labels: tuple[str, ...]
    question: dict[str, Any]

    @property
    def n(self) -> int:
        return len(self.labels)


def validate_task(task: dict[str, Any]) -> list[Decision]:
    decisions = task.get("decisions") or {}
    if not decisions:
        raise ConfigError("task.decisions is empty")
    out: list[Decision] = []
    for name, spec in decisions.items():
        labels = tuple(str(x) for x in spec.get("labels") or [])
        q = spec.get("question") or {}
        qtype = q.get("type")
        if qtype not in DECISION_TYPES:
            raise ConfigError(f"{name}: question.type must be one of {sorted(DECISION_TYPES)}, got {qtype!r}")
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ConfigError(f"{name}: needs >=2 distinct labels, got {labels}")
        crit = q.get("criteria")
        if qtype == "choice" and (not isinstance(crit, dict) or tuple(crit) != labels):
            raise ConfigError(f"{name}: choice criteria keys {list(crit or {})} must equal labels {list(labels)}")
        if qtype == "score":
            if not isinstance(crit, list) or len(crit) != len(labels):
                raise ConfigError(f"{name}: score needs one criterion per label")
            if labels != tuple(str(i) for i in range(len(labels))):
                raise ConfigError(f"{name}: score labels must be '0'..'{len(labels) - 1}'")
        if qtype == "noul" and labels != ("false", "true"):
            raise ConfigError(f"{name}: noul labels must be ['false', 'true']")
        out.append(Decision(name, qtype, labels, q))
    return out


def load_config(config_dir: Path = Path("configs"), profile: str | None = None,
                overrides: dict[str, Any] | None = None, project: str | None = None) -> dict[str, Any]:
    def _read(name: str) -> dict[str, Any]:
        with open(config_dir / name, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    cfg = _read("pipeline.yaml")
    profiles = cfg.pop("profiles", {}) or {}
    cfg["task"] = _read("task.yaml")
    cfg["model"] = _read("model.yaml")
    if project:
        # A goal-driven project (see spec.py): its task.yaml *replaces* the default task (merging
        # would leave the default decisions in), and project.yaml is an override layer.
        pdir = config_dir.parent / "projects" / project
        if not (pdir / "task.yaml").exists():
            raise ConfigError(f"project {project!r} has no {pdir / 'task.yaml'}; create it with `laya-tiny new`")
        with open(pdir / "task.yaml", encoding="utf-8") as f:
            cfg["task"] = yaml.safe_load(f)
        with open(pdir / "project.yaml", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
        cfg["project"] = project
        log.info("project %s loaded", project)
    if profile:
        if profile not in profiles:
            raise ConfigError(f"unknown profile {profile!r}; have {sorted(profiles)}")
        cfg = deep_merge(cfg, profiles[profile])
        log.info("profile %s applied", profile)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["profile"] = profile or "default"
    cfg.setdefault("project", None)
    validate_task(cfg["task"])
    enc = cfg["model"]["encoder"]
    if enc["d_model"] % enc["heads"]:
        raise ConfigError("model.encoder.d_model must be divisible by heads")
    return cfg


def decisions(cfg: dict[str, Any]) -> list[Decision]:
    return validate_task(cfg["task"])


def questions(cfg: dict[str, Any]) -> dict[str, Any]:
    """The Laya question dict, exactly as the teacher sees it."""
    return {d.name: d.question for d in decisions(cfg)}
