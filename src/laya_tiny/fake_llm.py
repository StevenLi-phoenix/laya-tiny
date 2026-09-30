"""Offline stand-in for the big model (tests, CI smoke). It designs the support-ticket triage task
(so the keyword `rules` teacher can label it) and writes templated texts that follow the requested
labels. Same interface as llm.LLM."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from .data import _T_CHURN, _T_DEPT, _T_URG
from .llm import LLMError, extract_json

FAKE_SPEC = {
    "name": "ticket-triage",
    "summary": "Route a support ticket, rate its urgency and flag churn threats.",
    "language": "en",
    "input_description": "a customer support ticket, one to five sentences",
    "styles": ["terse chat message", "polite email", "angry rant"],
    "decisions": [
        {"name": "department", "type": "choice", "instructions": "Which team should handle this ticket?",
         "options": {"billing": "invoices, charges, refunds", "technical": "bugs, crashes, outages",
                     "account": "login, password, profile", "sales": "pricing, plans, quotes"}},
        {"name": "urgency", "type": "score", "instructions": "How urgent is this ticket?",
         "levels": ["not urgent", "soon: broken with a workaround", "blocking right now"]},
        {"name": "churn_risk", "type": "noul", "instructions": "Does the customer threaten to leave?",
         "true": "says they will cancel or switch", "false": "no threat to leave"},
    ],
}


class FakeLLM:
    def __init__(self, cfg: dict[str, Any], cache_dir: Path) -> None:
        self.calls = 0
        self.cache_hits = 0

    def chat(self, messages: list[dict[str, str]], *, seed: int = 0, **kw: Any) -> str:
        self.calls += 1
        if "design fixed classification tasks" in messages[0]["content"]:
            return json.dumps(FAKE_SPEC)
        user = messages[-1]["content"]
        rng = random.Random(f"{seed}|{user}")
        k = int(re.search(r"Write (\d+) different", user).group(1)) if "Write" in user else 5
        dept = next((d for d in _T_DEPT if f"-> {d} " in user), rng.choice(list(_T_DEPT)))
        m = re.search(r"-> level (\d)", user)
        urg = int(m.group(1)) if m else 0
        churn = 1 if "-> yes" in user else 0
        texts = [f"{rng.choice(_T_DEPT[dept]).capitalize()}, {rng.choice(_T_URG[urg])} "
                 f"{rng.choice(_T_CHURN[churn])} (ref {rng.randint(100, 99999)})".strip() for _ in range(k)]
        return json.dumps({"texts": texts})

    def chat_json(self, messages: list[dict[str, str]], validate: Callable[[Any], Any], *, seed: int = 0,
                  attempts: int = 3, **kw: Any) -> Any:
        try:
            return validate(extract_json(self.chat(messages, seed=seed)))
        except (ValueError, TypeError, KeyError) as e:
            raise LLMError(str(e)) from e
