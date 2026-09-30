"""End to end on the smoke profile: offline, CPU, a few seconds."""
import json

import pytest

from laya_tiny.config import load_config
from laya_tiny.stages import ORDER, Ctx, run
from laya_tiny.util import read_jsonl

pytestmark = pytest.mark.integration


def test_smoke_pipeline_runs_and_caches(sandbox):
    ctx = Ctx(load_config(sandbox / "configs", "smoke"), sandbox)
    first = run(ctx, "package")
    assert list(first) == ORDER
    work = sandbox / "work-smoke"
    # label files: one full distribution per decision for every row
    for r in read_jsonl(work / "label" / "corpus.jsonl"):
        assert [len(r["probs"][k]) for k in ("department", "urgency", "churn_risk")] == [4, 3, 2]
    meta = json.loads((work / "export" / "meta.json").read_text())
    assert meta["equivalence"]["fp32_vs_torch"]["max_abs_diff"] < 1e-4
    res = json.loads((work / "eval" / "results.json").read_text())
    names = [b["name"] for b in res["backends"]]
    assert names[:2] == ["laya-tiny int8", "laya-tiny fp32"] and "TF-IDF + LogReg" in names
    assert set(res["acceptance"]["checks"]) >= {"size_int8_mb", "cpu_p50_ms", "accuracy/department"}
    reports = list((work / "reports").glob("*.md"))
    assert reports and (work / "reports" / "overview.png").stat().st_size > 10_000
    assert run(ctx, "package") == {}                      # everything cached
    assert list(run(ctx, "train", force={"train"})) == ["train"]
