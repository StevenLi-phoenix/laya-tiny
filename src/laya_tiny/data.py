"""Stage `data`: download public support-ticket corpora, clean, filter and de-duplicate them
into one unlabelled corpus. All source text is treated as untrusted input."""
from __future__ import annotations

import csv
import hashlib
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from .stages import Ctx
from .util import log, sha256_file, text_id, write_jsonl

csv.field_size_limit(10_000_000)

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏  ﻿]")
_TAG = re.compile(r"<[^>\n]{1,200}>")
_SPACES = re.compile(r"[ \t ]+")
_NEWLINES = re.compile(r"\n{3,}")
_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]{1,40}?)\s*\}\}")
_NONALNUM = re.compile(r"[^0-9a-z]+")

# Bitext keeps entities as {{Placeholder}}; fill them so the student never learns braces.
_FILLERS = {
    "order number": ["#48213", "ORD-99812", "order 5531"],
    "invoice number": ["INV-2044", "invoice 3391", "#77120"],
    "account type": ["Pro", "Business", "Free"],
    "account category": ["premium", "standard", "business"],
    "person name": ["Alex", "Maria", "Sam"],
    "refund amount": ["$25", "$140", "€60"],
    "money amount": ["$25", "$140", "€60"],
    "delivery city": ["Chicago", "Berlin", "Lyon"],
    "delivery country": ["Canada", "Spain", "Japan"],
    "currency symbol": ["$", "€", "£"],
}


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = _TAG.sub(" ", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NEWLINES.sub("\n\n", text).strip()


def truncate(text: str, max_chars: int) -> str:
    """Cut at a sentence (or word) boundary so teacher and student see exactly the same text."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n"):
        i = head.rfind(sep)
        if i >= max_chars * 0.5:
            return head[: i + 1].strip()
    i = head.rfind(" ")
    return (head[:i] if i > 0 else head).strip()


def dedup_key(text: str) -> str:
    return _NONALNUM.sub(" ", text.lower()).strip()


def looks_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters) >= 0.97


def fill_placeholders(text: str, seed: str) -> str:
    rng = random.Random(seed)

    def sub(m: re.Match[str]) -> str:
        key = m.group(1).strip().lower()
        options = _FILLERS.get(key)
        return rng.choice(options) if options else key
    return _PLACEHOLDER.sub(sub, text)


def download(url: str, cache_dir: Path) -> Path:
    dest = cache_dir / url.rsplit("/", 1)[-1]
    if not dest.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs from config
        tmp.rename(dest)
    log.info("source %s sha256=%s (%.1f MB)", dest.name, sha256_file(dest)[:16], dest.stat().st_size / 1e6)
    return dest


def _read_csv(path: Path) -> Iterator[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def read_tobi_bueck(path: Path, weak_map: dict[str, dict[str, str]], languages: list[str]) -> Iterator[dict[str, Any]]:
    for r in _read_csv(path):
        if languages and (r.get("language") or "").strip() not in languages:
            continue
        subject, body = (r.get("subject") or "").strip(), (r.get("body") or "").strip()
        if subject.lower() in ("", "nan"):
            subject = ""
        text = f"{subject}\n\n{body}" if subject else body
        yield {"text": text, "weak": dict(weak_map.get((r.get("queue") or "").strip(), {})),
               "meta": {"queue": r.get("queue"), "priority": r.get("priority")}}


def read_bitext(path: Path, weak_map: dict[str, dict[str, str]]) -> Iterator[dict[str, Any]]:
    for r in _read_csv(path):
        instr = r.get("instruction") or ""
        intent = (r.get("intent") or "").strip()
        yield {"text": fill_placeholders(instr, instr), "weak": dict(weak_map.get(intent, {})),
               "meta": {"intent": intent}}


# --- offline templated corpus (smoke tests / CI) ---------------------------------------------
_T_DEPT = {
    "billing": ["I was charged twice this month", "my invoice has the wrong amount", "the refund never arrived",
                "my card payment was declined", "the VAT on the bill is wrong"],
    "technical": ["the app crashes on startup", "the API returns 500 errors", "the dashboard is very slow",
                  "sync stopped working", "uploads fail with an error"],
    "account": ["I can't log in", "the password reset email never arrives", "please change my profile email",
                "I need admin permissions for a colleague", "2FA codes are rejected"],
    "sales": ["how much is the enterprise plan", "can we get a quote for 40 seats", "do you have a nonprofit discount",
              "we want to upgrade to Pro", "what does the Business plan include"],
}
_T_URG = [["", "just a quick question.", "no rush."],
          ["there's a workaround for now.", "we can manage for a few days.", "it's annoying but we cope."],
          ["nothing works right now!", "our whole team is blocked.", "we are losing money every minute."]]
_T_CHURN = [["", "thanks.", "please advise."],
            ["fix this or we cancel.", "we're switching to a competitor.", "we will not renew."]]


def generate_templates(n: int, seed: int) -> Iterator[dict[str, Any]]:
    rng = random.Random(seed)
    depts = list(_T_DEPT)
    for i in range(n):
        d = depts[i % len(depts)]
        u = rng.choice([0, 0, 1, 1, 2])
        c = 1 if rng.random() < 0.2 else 0
        parts = [rng.choice(_T_DEPT[d]).capitalize() + ",", rng.choice(_T_URG[u]), rng.choice(_T_CHURN[c]),
                 f"(ref {rng.randint(100, 99999)})"]
        yield {"text": " ".join(p for p in parts if p), "weak": {"department": d}, "meta": {"u": u, "c": c}}


def build_corpus(cfg: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dcfg = cfg["data"]
    cache_dir = root / dcfg["cache_dir"]
    weak = dcfg.get("weak_labels") or {}
    languages = list(dcfg.get("languages") or [])
    rng = random.Random(cfg["seed"])
    stats: dict[str, Counter[str]] = {}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for src in dcfg["sources"]:
        kind, name = src["kind"], src["name"]
        if kind == "tobi_bueck":
            it = read_tobi_bueck(download(src["url"], cache_dir), weak.get("tobi_bueck_queue", {}), languages)
        elif kind == "bitext":
            it = read_bitext(download(src["url"], cache_dir), weak.get("bitext_intent", {}))
        elif kind == "templates":
            it = generate_templates(int(src.get("rows", 500)), cfg["seed"])
        else:
            raise ValueError(f"unknown source kind {kind!r}")
        raw = list(it)
        rng.shuffle(raw)                       # so max_rows samples across intents instead of the file head
        st = stats.setdefault(name, Counter())
        kept_here = 0
        for r in raw:
            st["read"] += 1
            text = truncate(clean_text(r["text"]), int(dcfg["max_chars"]))
            if len(text) < int(dcfg["min_chars"]):
                st["too_short"] += 1
                continue
            if "en" in languages and not looks_english(text):
                st["not_english"] += 1
                continue
            key = dedup_key(text)
            if key in seen:
                st["duplicate"] += 1
                continue
            if src.get("max_rows") and kept_here >= int(src["max_rows"]):
                st["over_source_cap"] += 1
                continue
            seen.add(key)
            kept_here += 1
            st["kept"] += 1
            rows.append({"id": text_id(text), "text": text, "source": name, "weak": r["weak"]})
    cap = int(dcfg.get("max_rows") or 0)
    if cap and len(rows) > cap:
        rng.shuffle(rows)
        rows = rows[:cap]
    rows.sort(key=lambda r: r["id"])
    summary = {"rows": len(rows), "by_source": dict(Counter(r["source"] for r in rows)),
               "with_weak_label": sum(1 for r in rows if r["weak"]),
               "filters": {k: dict(v) for k, v in stats.items()}}
    return rows, summary


def run(ctx: Ctx) -> dict[str, Any]:
    rows, summary = build_corpus(ctx.cfg, ctx.root)
    write_jsonl(ctx.path("data", "corpus.jsonl"), rows)
    for name, st in summary["filters"].items():
        log.info("source %-16s %s", name, st)
    return summary


def stable_bucket(id_: str) -> float:
    """Deterministic value in [0, 1) from an id, for splits that survive re-ordering."""
    return int(hashlib.sha256(id_.encode()).hexdigest()[:8], 16) / 0x100000000
