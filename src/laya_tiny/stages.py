"""A tiny make: each stage declares inputs, config keys and outputs, and is skipped when the
fingerprint of (stage code, config subset, input file hashes) matches its last successful run."""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .util import log, sha256_bytes, sha256_file, stable_json, write_json


@dataclass
class Ctx:
    cfg: dict[str, Any]
    root: Path = field(default_factory=Path.cwd)

    @property
    def work(self) -> Path:
        return self.root / self.cfg["work_dir"]

    def path(self, *parts: str) -> Path:
        return self.work.joinpath(*parts)

    @property
    def holdout(self) -> Path:
        return self.root / self.cfg["holdout"]


@dataclass(frozen=True)
class Stage:
    name: str
    module: str                                   # laya_tiny.<module>.run(ctx) -> summary dict
    deps: tuple[str, ...]
    config_keys: tuple[str, ...]                   # dotted paths into the merged config
    outputs: Callable[[Ctx], list[Path]]
    extra_inputs: Callable[[Ctx], list[Path]] = lambda ctx: []


def _get(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _p(*parts: str) -> Callable[[Ctx], list[Path]]:
    return lambda ctx: [ctx.path(*parts)]


STAGES: dict[str, Stage] = {s.name: s for s in [
    Stage("data", "data", (), ("data", "seed"), _p("data", "corpus.jsonl")),
    Stage("label", "teacher", ("data",), ("task.teacher.kind", "task.teacher.checkpoint", "task.teacher.revision", "task.teacher.precision", "task.decisions"),
          lambda ctx: [ctx.path("label", "corpus.jsonl"), ctx.path("label", "holdout.jsonl")],
          lambda ctx: [ctx.holdout]),
    Stage("split", "split", ("data", "label"), ("data.leak_jaccard", "data.val_fraction", "seed"),
          lambda ctx: [ctx.path("split", "train.jsonl"), ctx.path("split", "val.jsonl")],
          lambda ctx: [ctx.holdout]),
    Stage("tokenizer", "tok", ("split",), ("model.tokenizer", "model.encoder.max_len"),
          _p("tokenizer", "tokenizer.json")),
    Stage("train", "train", ("split", "tokenizer"), ("model", "train", "task.decisions", "seed"),
          _p("train", "model.pt")),
    Stage("calibrate", "calibrate", ("split", "tokenizer", "train"), ("calibrate",),
          _p("calibrate", "calibration.json")),
    Stage("export", "export", ("tokenizer", "train", "calibrate"), ("export",),
          lambda ctx: [ctx.path("export", "student.onnx"), ctx.path("export", "student.int8.onnx"),
                       ctx.path("export", "meta.json")]),
    Stage("eval", "evaluate", ("split", "label", "export"), ("eval",),
          _p("eval", "results.json"), lambda ctx: [ctx.holdout]),
    Stage("report", "report", ("eval",), (), _p("report", "report.md")),
]}

ORDER = list(STAGES)


def plan(target: str) -> list[str]:
    """Stages needed for `target`, upstream first."""
    need: list[str] = []

    def visit(name: str) -> None:
        for dep in STAGES[name].deps:
            visit(dep)
        if name not in need:
            need.append(name)

    if target not in STAGES:
        raise KeyError(f"unknown stage {target!r}; have {ORDER}")
    visit(target)
    return need


def fingerprint(ctx: Ctx, stage: Stage) -> str:
    mod = importlib.import_module(f"laya_tiny.{stage.module}")
    inputs: list[Path] = [p for dep in stage.deps for p in STAGES[dep].outputs(ctx)] + stage.extra_inputs(ctx)
    payload = {
        "stage": stage.name,
        "code": sha256_file(Path(mod.__file__)),
        "config": {k: _get(ctx.cfg, k) for k in stage.config_keys},
        "inputs": {str(p.relative_to(ctx.root) if p.is_relative_to(ctx.root) else p):
                   (sha256_file(p) if p.exists() else None) for p in inputs},
    }
    return sha256_bytes(stable_json(payload).encode())


def stamp_path(ctx: Ctx, name: str) -> Path:
    return ctx.path(".stamps", f"{name}.json")


def is_fresh(ctx: Ctx, stage: Stage, fp: str) -> bool:
    import json

    sp = stamp_path(ctx, stage.name)
    if not sp.exists() or not all(p.exists() for p in stage.outputs(ctx)):
        return False
    return json.loads(sp.read_text()).get("fingerprint") == fp


def run(ctx: Ctx, target: str, force: set[str] | None = None, only: bool = False) -> dict[str, Any]:
    force = force or set()
    names = [target] if only else plan(target)
    summaries: dict[str, Any] = {}
    for name in names:
        stage = STAGES[name]
        fp = fingerprint(ctx, stage)
        if name not in force and "all" not in force and is_fresh(ctx, stage, fp):
            log.info("[%s] up to date (%s), skipping", name, fp[:10])
            continue
        missing = [p for dep in stage.deps for p in STAGES[dep].outputs(ctx) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"[{name}] missing upstream outputs: {missing}")
        log.info("[%s] running (fingerprint %s)", name, fp[:10])
        t0 = time.perf_counter()
        mod = importlib.import_module(f"laya_tiny.{stage.module}")
        summary = mod.run(ctx) or {}
        dt = time.perf_counter() - t0
        write_json(stamp_path(ctx, name), {"fingerprint": fp, "seconds": round(dt, 2),
                                            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "summary": summary})
        log.info("[%s] done in %.1fs %s", name, dt, stable_json(summary)[:300])
        summaries[name] = summary
    return summaries
