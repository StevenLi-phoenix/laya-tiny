"""Goal-driven flow: LLM spec validation, synthesis parsing, and `laya-tiny new` end to end (fake LLM)."""
import copy
import json

import pytest

from laya_tiny.fake_llm import FAKE_SPEC
from laya_tiny.llm import LLMError, extract_json
from laya_tiny.spec import normalise_spec, task_from_spec
from laya_tiny.synth import parse_texts, plan

pytestmark = pytest.mark.unit


def test_extract_json_tolerates_fences_and_chatter():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"texts": ["x"]} hope it helps') == {"texts": ["x"]}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_normalise_spec_converts_to_task_contract():
    spec = normalise_spec(copy.deepcopy(FAKE_SPEC))
    d = spec["decisions"]
    assert d["department"]["labels"] == ["billing", "technical", "account", "sales"]
    assert d["urgency"]["labels"] == ["0", "1", "2"] and d["churn_risk"]["labels"] == ["false", "true"]
    task = task_from_spec(spec, "goal")
    assert task["teacher"]["checkpoint"] == "typed-decisions" and task["goal"] == "goal"
    zh = task_from_spec({**spec, "language": "zh"}, "g")
    assert zh["teacher"]["checkpoint"] == "multilingual" and zh["teacher"]["revision"] is None


@pytest.mark.parametrize("mutate, msg", [
    (lambda s: s.update(decisions=[]), "1-4"),
    (lambda s: s["decisions"][0].update(name="Bad Name!"), "bad or duplicate"),
    (lambda s: s["decisions"][0].update(options={"only": "one"}), "2-6 options"),
    (lambda s: s["decisions"][1].update(levels=["a", "b"]), "3-5 levels"),
    (lambda s: s["decisions"][2].update(type="multi"), "type must be"),
    (lambda s: s["decisions"][0]["options"].update({"true": "x"}), "bad option key"),
])
def test_normalise_spec_rejects_bad_llm_output(mutate, msg):
    raw = copy.deepcopy(FAKE_SPEC)
    mutate(raw)
    with pytest.raises(ValueError, match=msg):
        normalise_spec(raw)


def test_parse_texts_filters_meta_and_empty():
    out = parse_texts({"texts": ["Text 1: hello there friend", "ok", "The app crashed again today.", 5,
                                 "Label: billing -- charged twice"]}, 5, 200)
    assert out == ["The app crashed again today."]
    with pytest.raises(ValueError):
        parse_texts({"texts": ["ok"]}, 5, 200)


def test_plan_covers_every_label_combination():
    from laya_tiny.config import validate_task

    decs = validate_task({"decisions": normalise_spec(copy.deepcopy(FAKE_SPEC))["decisions"]})
    reqs = plan(decs, ["a", "b"], rows=240, per_request=10, seed=1)
    combos = {tuple(r["labels"].values()) for r in reqs}
    assert len(reqs) == 24 and len(combos) == 4 * 3 * 2


@pytest.mark.integration
def test_new_goal_end_to_end_with_fake_llm(sandbox, monkeypatch):
    from laya_tiny.cli import main

    monkeypatch.chdir(sandbox)
    assert main(["--profile", "smoke-new", "new", "triage support tickets", "--name", "demo",
                 "--rows", "200", "--holdout-rows", "40"]) == 0
    pkg = sandbox / "projects" / "demo" / "work" / "package"
    meta = json.loads((pkg / "model.json").read_text())
    assert meta["goal"] == "triage support tickets" and set(meta["outputs"]) == {"department", "urgency", "churn_risk"}
    assert (pkg / "model.onnx").stat().st_size > 10_000 and (pkg / "predict.py").exists()
    assert (sandbox / "projects" / "demo" / "reports").glob("*.md")
    # second call reuses the designed project and every cached stage
    assert main(["--profile", "smoke-new", "new", "ignored", "--name", "demo", "--rows", "200",
                 "--holdout-rows", "40"]) == 0
