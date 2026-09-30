"""Stage `label`: run the teacher over the corpus and the holdout, keeping the full probability
distribution for every decision. Resumable: rows already in the per-teacher cache are never
re-computed, so an interrupted run continues where it stopped."""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol, Sequence

from .config import Decision, decisions, questions
from .stages import Ctx
from .util import iter_jsonl, log, pick_device, read_jsonl, sha256_bytes, stable_json, write_json, write_jsonl

Dist = dict[str, list[float]]


class TeacherError(ValueError):
    pass


def validate_dist(dec: Decision, probs: Sequence[float]) -> list[float]:
    """Teacher output is untrusted: check shape and finiteness, clip, renormalise."""
    if len(probs) != dec.n:
        raise TeacherError(f"{dec.name}: expected {dec.n} probabilities, got {len(probs)}")
    vals = [float(p) for p in probs]
    if not all(math.isfinite(p) and p >= -1e-6 for p in vals):
        raise TeacherError(f"{dec.name}: non-finite or negative probabilities {vals}")
    vals = [max(p, 1e-6) for p in vals]
    s = sum(vals)
    if not 0.5 < s < 1.5:
        raise TeacherError(f"{dec.name}: probabilities sum to {s:.3f}")
    return [round(p / s, 6) for p in vals]


def laya_answer_to_dist(dec: Decision, answer: dict[str, Any]) -> list[float]:
    """Map one Laya answer object to a distribution ordered like `dec.labels`."""
    if dec.type == "noul":
        p = float(answer["noul"])
        return validate_dist(dec, [1.0 - p, p])
    probs = answer.get("probabilities") or {}
    missing = [lab for lab in dec.labels if lab not in probs]
    if missing:
        raise TeacherError(f"{dec.name}: teacher answer lacks labels {missing}")
    return validate_dist(dec, [probs[lab] for lab in dec.labels])


class Teacher(Protocol):
    def identity(self) -> dict[str, Any]: ...
    def predict(self, texts: list[str]) -> list[Dist]: ...


class LayaTeacher:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.tcfg = cfg["task"]["teacher"]
        self.decs = decisions(cfg)
        self.questions = questions(cfg)
        self._router: Any = None

    def _load(self) -> Any:
        if self._router is None:
            import laya

            device = pick_device(self.tcfg.get("device", "auto"))
            t0 = time.perf_counter()
            self._router = laya.Router(device=device, revision=self.tcfg.get("revision"))
            self._router.predict("warm-up", self.questions, model=self.tcfg["checkpoint"])
            log.info("teacher %s loaded on %s in %.1fs", self.tcfg["checkpoint"], device, time.perf_counter() - t0)
        return self._router

    def agent(self) -> Any:
        return self._load()._agents[self.tcfg["checkpoint"]]

    def identity(self) -> dict[str, Any]:
        import laya

        a = self.agent()
        return {"kind": "laya", "laya_version": getattr(laya, "__version__", "?"), "checkpoint": self.tcfg["checkpoint"],
                "repo": a.model_id_or_path, "subfolder": a.subfolder, "revision": a.revision,
                "temperature": list(a.temperature) if a.temperature is not None else None,
                "temperature_by_options": a.temperature_by_options}

    def predict(self, texts: list[str]) -> list[Dist]:
        router = self._load()
        reqs = [{"state": t, "questions": self.questions, "model": self.tcfg["checkpoint"]} for t in texts]
        out = router.predict_batch(reqs, batch_size=int(self.tcfg.get("batch_size", 16)), sort_by_length=True)
        return [{d.name: laya_answer_to_dist(d, r["answers"][d.name]) for d in self.decs} for r in out]


# Keyword teacher: deterministic, offline, instant. Only for smoke runs and tests.
_RULES: dict[str, dict[str, list[str]]] = {
    "department": {
        "billing": ["charge", "invoice", "refund", "payment", "card", "bill", "vat", "paid"],
        "technical": ["crash", "error", "slow", "bug", "api", "sync", "upload", "down", "500"],
        "account": ["log in", "login", "password", "profile", "permission", "admin", "2fa", "account"],
        "sales": ["price", "plan", "quote", "upgrade", "discount", "seats", "enterprise"],
    },
    "urgency": {
        "0": ["question", "no rush", "curious", "how do"],
        "1": ["workaround", "manage", "cope", "annoying"],
        "2": ["blocked", "right now", "losing money", "nothing works", "down", "urgent"],
    },
    "churn_risk": {
        "false": [],
        "true": ["cancel", "competitor", "not renew", "switching", "leave", "leaving"],
    },
}


class RulesTeacher:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.decs = decisions(cfg)

    def identity(self) -> dict[str, Any]:
        return {"kind": "rules", "version": 1, "rules_hash": sha256_bytes(stable_json(_RULES).encode())[:12]}

    def predict(self, texts: list[str]) -> list[Dist]:
        out = []
        for t in texts:
            low = t.lower()
            row: Dist = {}
            for d in self.decs:
                table = _RULES.get(d.name, {})
                scores = [sum(1 for kw in table.get(lab, []) if re.search(r"\b" + re.escape(kw), low))
                          for lab in d.labels]
                if d.type == "noul":
                    scores[0] = 0.5          # "no" wins unless a keyword fires
                logits = [1.5 * s for s in scores]
                m = max(logits)
                ex = [math.exp(v - m) for v in logits]
                row[d.name] = validate_dist(d, [e / sum(ex) for e in ex])
            out.append(row)
        return out


def make_teacher(cfg: dict[str, Any]) -> Teacher:
    kind = cfg["task"]["teacher"]["kind"]
    if kind == "laya":
        return LayaTeacher(cfg)
    if kind == "rules":
        return RulesTeacher(cfg)
    raise ValueError(f"unknown teacher kind {kind!r}")


def teacher_key(identity: dict[str, Any], qs: dict[str, Any]) -> str:
    return sha256_bytes(stable_json({"teacher": identity, "questions": qs}).encode())[:12]


def load_cache(path: Path) -> dict[str, Dist]:
    cache: dict[str, Dist] = {}
    if not path.exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                cache[row["id"]] = row["probs"]
            except (json.JSONDecodeError, KeyError):
                log.warning("label cache %s line %d is corrupt (interrupted write?), ignoring", path.name, n)
    return cache


def label_rows(teacher: Teacher, rows: list[dict[str, Any]], cache_path: Path, chunk: int = 256) -> dict[str, Dist]:
    cache = load_cache(cache_path)
    todo = [r for r in rows if r["id"] not in cache]
    log.info("labelling %d rows (%d cached) -> %s", len(todo), len(rows) - len(todo), cache_path.name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    done = 0
    with open(cache_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), chunk):
            batch = todo[i:i + chunk]
            preds = teacher.predict([r["text"] for r in batch])
            if len(preds) != len(batch):
                raise TeacherError(f"teacher returned {len(preds)} results for {len(batch)} texts")
            for r, p in zip(batch, preds):
                cache[r["id"]] = p
                f.write(json.dumps({"id": r["id"], "probs": p}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            done += len(batch)
            rate = done / max(time.perf_counter() - t0, 1e-9)
            log.info("labelled %d/%d  %.1f rows/s  eta %.1f min", done, len(todo), rate,
                     (len(todo) - done) / max(rate, 1e-9) / 60)
    return cache


def run(ctx: Ctx) -> dict[str, Any]:
    cfg = ctx.cfg
    teacher = make_teacher(cfg)
    ident = teacher.identity()
    key = teacher_key(ident, questions(cfg))
    cache_path = ctx.path("label", "cache", f"{ident['kind']}-{key}.jsonl")
    corpus = read_jsonl(ctx.path("data", "corpus.jsonl"))
    holdout = read_jsonl(ctx.holdout)
    t0 = time.perf_counter()
    cache = label_rows(teacher, holdout + corpus, cache_path)
    dt = time.perf_counter() - t0
    for name, rows in (("corpus", corpus), ("holdout", holdout)):
        write_jsonl(ctx.path("label", f"{name}.jsonl"), ({"id": r["id"], "probs": cache[r["id"]]} for r in rows))
    meta = {"teacher": ident, "teacher_key": key, "questions": questions(cfg), "rows": len(corpus),
            "holdout_rows": len(holdout), "seconds": round(dt, 1)}
    write_json(ctx.path("label", "meta.json"), meta)
    return {"teacher_key": key, "rows": len(corpus), "holdout_rows": len(holdout), "seconds": round(dt, 1)}


def read_labels(path: Path) -> dict[str, Dist]:
    return {r["id"]: r["probs"] for r in iter_jsonl(path)}
