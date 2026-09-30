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
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+\s*[.)])\s*")


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


def _keep(items: list[Any], k: int, max_chars: int) -> list[str]:
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


def parse_texts(raw: Any, k: int, max_chars: int) -> list[str]:
    items = raw.get("texts") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError('expected {"texts": [...]}')
    return _keep(items, k, max_chars)


def plain_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Same request, but one text per line instead of JSON: texts full of code and quotes are where a
    small local model breaks JSON escaping, and plain lines need no escaping at all."""
    system = messages[0]["content"].replace("Reply with JSON only.", "Reply with the texts only.")
    user = messages[1]["content"].rsplit("Return {", 1)[0]
    return [{"role": "system", "content": system},
            {"role": "user", "content": user + "Return exactly one text per line: no numbering, no quotes, "
                                               "no blank lines, no commentary, and no line breaks inside a text."}]


def parse_lines(content: str, k: int, max_chars: int) -> list[str]:
    lines = [_BULLET.sub("", ln).strip().strip('"').strip() for ln in (content or "").splitlines()]
    return _keep([ln for ln in lines if ln], k, max_chars)


def synth_one(llm: Any, spec: dict[str, Any], decs: list[Decision], req: dict[str, Any], purpose: str, *,
              max_chars: int, seed: int, temperature: float) -> list[dict[str, Any]]:
    msgs = build_messages(spec, decs, req, purpose)
    req_seed = seed * 100_003 + req["i"]
    meta: dict[str, Any] = {"style": req["style"], "req": req["i"]}
    try:
        texts = llm.chat_json(msgs, lambda raw: parse_texts(raw, req["k"], max_chars),
                              seed=req_seed, temperature=temperature)
    except LLMError as e:
        log.warning("synth request %d: JSON failed (%s), falling back to plain lines", req["i"], str(e)[:120])
        try:
            content = llm.chat(plain_messages(msgs), seed=req_seed + 7, temperature=temperature, json_mode=False)
            texts = parse_lines(content, req["k"], max_chars)
        except (LLMError, ValueError) as e2:
            log.warning("synth request %d failed, skipped: %s", req["i"], str(e2)[:160])
            return []
        meta["fallback"] = "lines"
        log.info("synth request %d recovered %d texts via plain-lines fallback", req["i"], len(texts))
    return [{"text": t, "weak": dict(req["labels"]), "meta": dict(meta)} for t in texts]


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

    temperature = float(scfg.get("temperature", 0.95))

    def one(req: dict[str, Any]) -> list[dict[str, Any]]:
        return synth_one(llm, spec, decs, req, purpose, max_chars=max_chars, seed=seed, temperature=temperature)

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
