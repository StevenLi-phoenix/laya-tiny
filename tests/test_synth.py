"""synth: when the big model keeps emitting broken JSON (e.g. code with unescaped quotes), the request
falls back to one-text-per-line plain output instead of being dropped."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from laya_tiny.fake_llm import FAKE_SPEC
from laya_tiny.llm import LLM
from laya_tiny.synth import parse_lines, synth_one


class _Decision:
    def __init__(self, name: str) -> None:
        self.name = name
        self.type = "noul"
        self.question = {"instructions": "Is it a bug?", "criteria": {"true": "a defect", "false": "not a defect"}}


BROKEN = '{"texts": ["It fails on `print("hi")` with a traceback", "second"]}'  # unescaped quotes
LINES = ('1. It fails on `print("hi")` with a traceback every single time\n'
         '- "Crash when saving a file with unicode name"\n'
         '\n'
         'Label: bug\n'
         '* The docs page for install is out of date and misleading')


class _StubLLM(LLM):
    def __init__(self, cache_dir: Path, plain: str) -> None:
        super().__init__({"base_url": "http://stub/v1"}, cache_dir)
        self.plain = plain
        self.modes: list[bool] = []

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        json_mode = "response_format" in payload
        self.modes.append(json_mode)
        content = BROKEN if json_mode else self.plain
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def _req() -> dict[str, Any]:
    return {"i": 3, "labels": {"is_bug": "true"}, "style": "terse", "length": "one short sentence", "k": 5}


def test_parse_lines_strips_bullets_quotes_and_meta() -> None:
    out = parse_lines(LINES, k=5, max_chars=400)
    assert out == ['It fails on `print("hi")` with a traceback every single time',
                   "Crash when saving a file with unicode name",
                   "The docs page for install is out of date and misleading"]


def test_broken_json_falls_back_to_plain_lines(tmp_path: Path) -> None:
    llm = _StubLLM(tmp_path, LINES)
    rows = synth_one(llm, FAKE_SPEC, [_Decision("is_bug")], _req(), "train", max_chars=400, seed=1, temperature=0.9)
    assert llm.modes == [True, True, True, False]          # 3 JSON attempts, then one plain request
    assert len(rows) == 3
    assert rows[0]["weak"] == {"is_bug": "true"} and rows[0]["meta"]["fallback"] == "lines"


def test_fallback_with_nothing_usable_is_skipped(tmp_path: Path) -> None:
    llm = _StubLLM(tmp_path, "Label: bug\nok")
    assert synth_one(llm, FAKE_SPEC, [_Decision("is_bug")], _req(), "train", max_chars=400, seed=1,
                     temperature=0.9) == []
