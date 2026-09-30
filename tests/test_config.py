import copy

import pytest

from laya_tiny.config import ConfigError, deep_merge, load_config, questions, validate_task

pytestmark = pytest.mark.unit


def test_default_and_smoke_profiles_load(repo_root):
    base = load_config(repo_root / "configs")
    smoke = load_config(repo_root / "configs", "smoke")
    assert base["task"]["teacher"]["kind"] == "laya"
    assert smoke["task"]["teacher"]["kind"] == "rules"
    assert smoke["task"]["teacher"]["checkpoint"] == "typed-decisions"      # merged, not replaced
    assert smoke["data"]["sources"] == [{"name": "templates", "kind": "templates", "rows": 700}]  # lists replace


def test_questions_match_laya_skill_contract(repo_root):
    q = questions(load_config(repo_root / "configs"))
    assert list(q) == ["department", "urgency", "churn_risk"]
    assert list(q["department"]["criteria"]) == ["billing", "technical", "account", "sales"]
    assert len(q["urgency"]["criteria"]) == 3
    assert q["churn_risk"]["type"] == "noul"


def test_deep_merge_does_not_mutate():
    a = {"x": {"y": 1, "z": [1, 2]}}
    b = {"x": {"z": [3]}}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 1, "z": [3]}} and a == {"x": {"y": 1, "z": [1, 2]}}


@pytest.mark.parametrize("mutate, msg", [
    (lambda t: t["decisions"]["department"].update(labels=["billing", "technical"]), "criteria keys"),
    (lambda t: t["decisions"]["churn_risk"].update(labels=["no", "yes"]), "noul labels"),
    (lambda t: t["decisions"]["urgency"].update(labels=["low", "mid", "high"]), "score labels"),
    (lambda t: t["decisions"]["urgency"]["question"].update(type="rank"), "question.type"),
])
def test_validate_task_rejects_inconsistent_contracts(repo_root, mutate, msg):
    task = copy.deepcopy(load_config(repo_root / "configs")["task"])
    mutate(task)
    with pytest.raises(ConfigError, match=msg):
        validate_task(task)


def test_unknown_profile(repo_root):
    with pytest.raises(ConfigError):
        load_config(repo_root / "configs", "nope")


def test_questions_json_mirrors_task_yaml(repo_root):
    """tools/laya_onnx_predict.mjs reads configs/questions.json; it must not drift from task.yaml."""
    import json

    on_disk = json.loads((repo_root / "configs" / "questions.json").read_text())
    assert on_disk == questions(load_config(repo_root / "configs"))


def test_cli_overrides_parse_yaml_values():
    from laya_tiny.cli import parse_overrides

    assert parse_overrides(["data.max_rows=4000", "train.device=cpu", "eval.external=[]"]) == {
        "data": {"max_rows": 4000}, "train": {"device": "cpu"}, "eval": {"external": []}}
    with pytest.raises(SystemExit):
        parse_overrides(["novalue"])
