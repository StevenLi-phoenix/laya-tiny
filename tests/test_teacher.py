import json
import math

import pytest

from laya_tiny.config import decisions, questions
from laya_tiny.teacher import (RulesTeacher, TeacherError, label_rows, laya_answer_to_dist, load_cache,
                               teacher_key, validate_dist)

pytestmark = pytest.mark.unit

LAYA_ANSWERS = {  # shape copied from a real typed-decisions response
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.7816, "technical": 0.0777, "account": 0.0662, "sales": 0.0746}},
    "urgency": {"type": "score", "score": 1.7654, "probabilities": {"0": 0.0395, "1": 0.1556, "2": 0.8049}},
    "churn_risk": {"type": "noul", "noul": 0.7682},
}


def test_laya_answer_maps_to_ordered_distributions(smoke_cfg):
    d = {x.name: x for x in decisions(smoke_cfg)}
    assert laya_answer_to_dist(d["department"], LAYA_ANSWERS["department"])[0] == pytest.approx(0.7816, abs=1e-3)
    assert laya_answer_to_dist(d["urgency"], LAYA_ANSWERS["urgency"])[2] == pytest.approx(0.8049, abs=1e-3)
    assert laya_answer_to_dist(d["churn_risk"], LAYA_ANSWERS["churn_risk"]) == pytest.approx([0.2318, 0.7682], abs=1e-4)
    with pytest.raises(TeacherError, match="lacks labels"):
        laya_answer_to_dist(d["department"], {"probabilities": {"billing": 1.0}})


def test_validate_dist_rejects_garbage(smoke_cfg):
    dep = decisions(smoke_cfg)[0]
    assert sum(validate_dist(dep, [0.2, 0.2, 0.2, 0.3])) == pytest.approx(1.0, abs=1e-5)
    for bad in ([0.5, 0.5], [math.nan, 0.5, 0.2, 0.3], [-0.5, 1.0, 0.3, 0.2], [2.0, 2.0, 2.0, 2.0]):
        with pytest.raises(TeacherError):
            validate_dist(dep, bad)


def test_rules_teacher_output_format(smoke_cfg):
    out = RulesTeacher(smoke_cfg).predict(["I was charged twice, refund or we cancel", "hello"])
    assert len(out) == 2
    for row in out:
        for d in decisions(smoke_cfg):
            assert len(row[d.name]) == d.n and sum(row[d.name]) == pytest.approx(1.0, abs=1e-5)
    assert max(range(4), key=lambda i: out[0]["department"][i]) == 0
    assert out[0]["churn_risk"][1] > 0.5 > out[1]["churn_risk"][1]


class CountingTeacher(RulesTeacher):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.seen: list[str] = []

    def predict(self, texts):
        self.seen.extend(texts)
        return super().predict(texts)


def test_label_rows_is_resumable_and_tolerates_torn_writes(smoke_cfg, tmp_path):
    rows = [{"id": f"r{i}", "text": f"ticket {i} refund"} for i in range(10)]
    cache = tmp_path / "cache.jsonl"
    t1 = CountingTeacher(smoke_cfg)
    label_rows(t1, rows[:6], cache, chunk=4)
    with open(cache, "a") as f:
        f.write('{"id": "r9", "pro')                      # simulated crash mid-line
    assert set(load_cache(cache)) == {f"r{i}" for i in range(6)}
    t2 = CountingTeacher(smoke_cfg)
    out = label_rows(t2, rows, cache, chunk=4)
    assert t2.seen == [f"ticket {i} refund" for i in range(6, 10)]
    assert set(out) == {r["id"] for r in rows}
    assert out["r0"] == json.loads(cache.read_text().splitlines()[0])["probs"]


def test_teacher_key_tracks_questions(smoke_cfg):
    ident = RulesTeacher(smoke_cfg).identity()
    q = questions(smoke_cfg)
    q2 = json.loads(json.dumps(q))
    q2["urgency"]["instructions"] = "How bad is it?"
    assert teacher_key(ident, q) == teacher_key(ident, questions(smoke_cfg))
    assert teacher_key(ident, q) != teacher_key(ident, q2)
