"""Goal -> task contract. The big model turns one sentence of intent into typed decisions
(the same shape as configs/task.yaml), an input description and writing styles for synthesis.

Written to projects/<name>/{goal.txt, spec.json, task.yaml, project.yaml}; everything after this
is the ordinary pipeline run with `--project <name>`."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError, validate_task
from .data import clean_text
from .llm import make_llm
from .util import log, write_json

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_LABEL = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,31}$")
LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"

SPEC_SYSTEM = """You design fixed classification tasks for a tiny text model.
Given a user's goal, decide what the model reads and which 1-4 decisions it must output.
Decision types:
- "choice": pick exactly one of 2-6 mutually exclusive options (a category).
- "score": an ordered scale of 3-5 levels, lowest first (severity, urgency, quality...).
- "noul": a yes/no question.
Rules: decisions must be answerable from the text alone; options must be mutually exclusive and cover
typical inputs (add an "other" option if needed); names are snake_case ASCII; option keys are short
lowercase ASCII identifiers; every option/level gets a one-line description.
Reply with JSON only."""

SPEC_SCHEMA_HINT = """{
  "name": "short-kebab-case-project-name",
  "summary": "one sentence: what the model does",
  "language": "ISO 639-1 code of the input texts, e.g. en or zh",
  "input_description": "what one input text looks like (who writes it, where, typical length)",
  "styles": ["5-10 distinct writing styles/situations to cover, e.g. 'terse one-line chat message'"],
  "decisions": [
    {"name": "snake_case", "type": "choice", "instructions": "question?",
     "options": {"key": "description", "...": "..."}},
    {"name": "snake_case", "type": "score", "instructions": "question?",
     "levels": ["lowest: description", "...", "highest: description"]},
    {"name": "snake_case", "type": "noul", "instructions": "yes/no question?",
     "true": "when the answer is yes", "false": "when the answer is no"}
  ]
}"""


def _s(v: Any, field: str, max_len: int = 300) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{field} must be a string")
    t = clean_text(v)
    if not t:
        raise ValueError(f"{field} is empty")
    return t[:max_len]


def normalise_spec(raw: Any) -> dict[str, Any]:
    """Validate the LLM's spec and convert it to our task contract. Raises ValueError with a reason."""
    if not isinstance(raw, dict):
        raise ValueError("top level must be an object")
    name = re.sub(r"[^a-z0-9-]+", "-", _s(raw.get("name", "task"), "name", 40).lower()).strip("-") or "task"
    lang = _s(raw.get("language", "en"), "language", 8).lower()[:2]
    decs = raw.get("decisions")
    if not isinstance(decs, list) or not 1 <= len(decs) <= 4:
        raise ValueError("decisions must be a list of 1-4 items")
    out: dict[str, Any] = {}
    for d in decs:
        if not isinstance(d, dict):
            raise ValueError("each decision must be an object")
        dname = _s(d.get("name"), "decision.name", 32).lower()
        if not _NAME.match(dname) or dname in out:
            raise ValueError(f"bad or duplicate decision name {dname!r} (snake_case, 2-32 chars)")
        qtype = d.get("type")
        instr = _s(d.get("instructions"), f"{dname}.instructions", 200)
        if qtype == "choice":
            opts = d.get("options")
            if not isinstance(opts, dict) or not 2 <= len(opts) <= 6:
                raise ValueError(f"{dname}: choice needs 2-6 options")
            crit: dict[str, str] = {}
            for k, v in opts.items():
                key = re.sub(r"\s+", "_", str(k).strip().lower())
                if not _LABEL.match(key) or key in ("true", "false") or key in crit:
                    raise ValueError(f"{dname}: bad option key {k!r}")
                crit[key] = _s(v, f"{dname}.{key}", 160)
            out[dname] = {"labels": list(crit), "question": {"type": "choice", "instructions": instr, "criteria": crit}}
        elif qtype == "score":
            levels = d.get("levels")
            if not isinstance(levels, list) or not 3 <= len(levels) <= 5:
                raise ValueError(f"{dname}: score needs 3-5 levels")
            crit_l = [_s(v, f"{dname}.level", 160) for v in levels]
            out[dname] = {"labels": [str(i) for i in range(len(crit_l))],
                          "question": {"type": "score", "instructions": instr, "criteria": crit_l}}
        elif qtype == "noul":
            out[dname] = {"labels": ["false", "true"],
                          "question": {"type": "noul", "instructions": instr,
                                       "criteria": {"true": _s(d.get("true"), f"{dname}.true", 160),
                                                    "false": _s(d.get("false"), f"{dname}.false", 160)}}}
        else:
            raise ValueError(f"{dname}: type must be choice, score or noul, got {qtype!r}")
    styles = [clean_text(s)[:120] for s in (raw.get("styles") or []) if isinstance(s, str) and clean_text(s)]
    spec = {
        "name": name,
        "summary": _s(raw.get("summary", name), "summary", 300),
        "language": lang if re.fullmatch(r"[a-z]{2}", lang) else "en",
        "input_description": _s(raw.get("input_description"), "input_description", 400),
        "styles": styles[:10] or ["short casual message", "detailed formal message", "angry rant", "polite question"],
        "decisions": out,
    }
    validate_task({"decisions": out})          # same contract check as a hand-written task.yaml
    return spec


def task_from_spec(spec: dict[str, Any], goal: str) -> dict[str, Any]:
    english = spec["language"] == "en"
    return {
        "version": 1,
        "name": spec["name"],
        "goal": goal,
        "summary": spec["summary"],
        "teacher": {"kind": "laya", "checkpoint": "typed-decisions" if english else "multilingual",
                    "repo": LAYA_REPO if english else "convaiinnovations/laya-multilingual",
                    "revision": LAYA_REVISION if english else None, "device": "auto", "precision": "fp32",
                    "batch_size": 8},
        "decisions": copy.deepcopy(spec["decisions"]),
    }


def project_overrides(name: str, spec: dict[str, Any], rows: int, holdout_rows: int) -> dict[str, Any]:
    base = f"projects/{name}"
    return {
        "work_dir": f"{base}/work",
        "runs_dir": f"{base}/runs",
        "reports_dir": f"{base}/reports",
        "holdout": f"{base}/work/data/holdout.jsonl",
        "data": {
            "languages": [spec["language"]],
            "max_rows": rows,
            "sources": [{"name": "llm-synth", "kind": "llm_synth", "rows": rows}],
            "holdout_synth": {"rows": holdout_rows},
        },
        "train": {"hard_weight": 0.3},
        "eval": {"external": [], "laya_latency_rows": 20},
    }


def design(goal: str, name: str | None, cfg: dict[str, Any], root: Path, rows: int, holdout_rows: int,
           force: bool = False) -> str:
    goal = clean_text(goal)
    if not goal:
        raise ConfigError("goal is empty")
    llm = make_llm(cfg["llm"], root / "projects" / ".llm-cache")
    if name and (root / "projects" / name / "task.yaml").exists() and not force:
        log.info("project %s already designed, reusing (pass --force-spec to redo)", name)
        return name
    messages = [{"role": "system", "content": SPEC_SYSTEM},
                {"role": "user", "content": f"Goal: {goal}\n\nReturn JSON shaped like:\n{SPEC_SCHEMA_HINT}"}]
    spec = llm.chat_json(messages, normalise_spec, seed=int(cfg["seed"]), temperature=0.3)
    name = name or spec["name"]
    pdir = root / "projects" / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "goal.txt").write_text(goal + "\n", encoding="utf-8")
    write_json(pdir / "spec.json", spec)
    task = task_from_spec(spec, goal)
    (pdir / "task.yaml").write_text("# Designed by the LLM from goal.txt; edit freely, then re-run.\n"
                                    + yaml.safe_dump(task, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (pdir / "project.yaml").write_text(yaml.safe_dump(project_overrides(name, spec, rows, holdout_rows),
                                                      allow_unicode=True, sort_keys=False), encoding="utf-8")
    log.info("designed project %s: %s", name, ", ".join(f"{k}[{len(v['labels'])}]" for k, v in task["decisions"].items()))
    return name
