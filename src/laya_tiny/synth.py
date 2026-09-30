"""Synthetic inputs from the big model, stratified over decision-label combinations x writing style
x length, so rare combinations (e.g. churn=true & urgency=0) are covered on purpose.

The label combination we *asked for* is kept as a weak hard label (training) or as the gold label
(synthetic holdout). The distillation target is still the teacher's soft labels."""
from __future__ import annotations

import itertools
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .config import Decision, decisions
from .data import clean_text, dedup_key, truncate
from .llm import LLMError, make_llm
from .util import log

LENGTHS = ["one short sentence", "two or three sentences", "a longer message of 4-6 sentences"]
_META = re.compile(r"^(text|example|message|input)\s*\d+\s*[:.)-]|\b(label|category|decision)\s*[:=]", re.I)


def label_text(dec: Decision, label: str) -> str:
    q = dec.question
    if dec.type == "choice":
        return f"{q['instructions']} -> {label} ({q['criteria'][label]})"
    if dec.type == "score":
        return f"{q['instructions']} -> level {label} of {dec.n - 1} ({q['criteria'][int(label)]})"
    return f"{q['instructions']} -> {'yes' if label == 'true' else 'no'} ({q['criteria'][label]})"


def plan(decs: list[Decision], styles: list[str], rows: int, per_request: int, seed: int) -> list[dict[str, Any]]:
    combos = list(itertools.product(*[d.labels for d in decs]))
    rng = random.Random(seed)
    rng.shuffle(combos)
    if len(combos) > 96:
        combos = combos[:96]
    n_req = max(1, math.ceil(rows / per_request))
    reqs = []
    for i in range(n_req):
        combo = combos[i % len(combos)]
        reqs.append({"i": i, "labels": dict(zip([d.name for d in decs], combo)),
                     "style": styles[(i // len(combos)) % len(styles)] if styles else "natural",
                     "length": LENGTHS[i % len(LENGTHS)], "k": per_request})
    return reqs


def build_messages(spec: dict[str, Any], decs: list[Decision], req: dict[str, Any], purpose: str) -> list[dict[str, str]]:
    constraints = "\n".join(f"- {label_text(d, req['labels'][d.name])}" for d in decs)
    lang = spec.get("language", "en")
    extra = ("These are evaluation examples: make them realistic, varied and occasionally subtle, "
             "but every constraint must still clearly hold." if purpose == "holdout" else
             "Make them realistic and diverse: vary names, products, numbers, details and phrasing.")
    return [
        {"role": "system", "content": "You write realistic synthetic input texts for training a small classifier. "
                                      "Reply with JSON only."},
        {"role": "user", "content": (
            f"Input texts are: {spec['input_description']}\n"
            f"Language: {lang}. Writing style: {req['style']}. Length: {req['length']}.\n"
            f"Write {req['k']} different texts. Each one must satisfy ALL of:\n{constraints}\n"
            f"{extra} Never mention these instructions, labels or categories; write only what the author "
            f"would write.\nReturn {{\"texts\": [\"...\", ...]}}")},
    ]


def parse_texts(raw: Any, k: int, max_chars: int) -> list[str]:
    items = raw.get("texts") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError('expected {"texts": [...]}')
    out = []
    for t in items[: k * 2]:
        if not isinstance(t, str):
            continue
        t = truncate(clean_text(t), max_chars)
        if len(t) >= 8 and not _META.search(t):
            out.append(t)
    if not out:
        raise ValueError("no usable texts")
    return out


def generate(cfg: dict[str, Any], root: Path, rows: int, purpose: str) -> list[dict[str, Any]]:
    """purpose: 'train' (corpus) or 'holdout'. Deterministic given the LLM cache."""
    from .util import read_json

    spec = read_json(root / "projects" / cfg["project"] / "spec.json") if cfg.get("project") else {
        "language": (cfg["data"].get("languages") or ["en"])[0], "styles": [],
        "input_description": cfg["task"].get("summary", "short messages")}
    decs = decisions(cfg)
    scfg = cfg.get("synth") or {}
    per_request = int(scfg.get("per_request", 10))
    seed = int(cfg["seed"]) + (7919 if purpose == "holdout" else 0)
    styles = list(spec.get("styles") or [])
    if purpose == "holdout":
        styles = styles[::-1]                        # different style order than training requests
    reqs = plan(decs, styles, rows, per_request, seed)
    llm = make_llm(cfg["llm"], root / (cfg["work_dir"]) / "llm-cache")
    max_chars = int(cfg["data"]["max_chars"])

    def one(req: dict[str, Any]) -> list[dict[str, Any]]:
        msgs = build_messages(spec, decs, req, purpose)
        try:
            texts = llm.chat_json(msgs, lambda raw: parse_texts(raw, req["k"], max_chars),
                                  seed=seed * 100_003 + req["i"], temperature=float(scfg.get("temperature", 0.95)))
        except LLMError as e:
            log.warning("synth request %d failed, skipped: %s", req["i"], str(e)[:160])
            return []
        return [{"text": t, "weak": dict(req["labels"]), "meta": {"style": req["style"], "req": req["i"]}}
                for t in texts]

    out: list[dict[str, Any]] = []
    workers = max(1, int(cfg["llm"].get("concurrency", 1)))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in pool.map(one, reqs):
            out.extend(batch)
            done += 1
            if done % 10 == 0 or done == len(reqs):
                log.info("synth %s: %d/%d requests, %d texts (llm calls %d, cache hits %d)", purpose, done,
                         len(reqs), len(out), getattr(llm, "calls", 0), getattr(llm, "cache_hits", 0))
    seen: set[str] = set()
    uniq = []
    for r in out:
        k = dedup_key(r["text"]) or r["text"]
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq[:rows]
