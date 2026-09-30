import pytest

from laya_tiny.config import load_config
from laya_tiny.stages import ORDER, STAGES, Ctx, fingerprint, plan

pytestmark = pytest.mark.unit


def test_plan_is_topological():
    full = plan("package")
    assert full == ORDER
    for name in full:
        for dep in STAGES[name].deps:
            assert full.index(dep) < full.index(name)
    assert plan("tokenizer") == ["data", "label", "split", "tokenizer"]


def test_fingerprint_tracks_only_declared_config(sandbox):
    base = load_config(sandbox / "configs", "smoke")
    ctx = Ctx(base, sandbox)
    fp = fingerprint(ctx, STAGES["train"])
    unrelated = Ctx(load_config(sandbox / "configs", "smoke", {"eval": {"latency_runs": 7}}), sandbox)
    related = Ctx(load_config(sandbox / "configs", "smoke", {"train": {"lr": 0.5}}), sandbox)
    assert fingerprint(unrelated, STAGES["train"]) == fp
    assert fingerprint(related, STAGES["train"]) != fp


def test_fingerprint_tracks_input_files(sandbox):
    ctx = Ctx(load_config(sandbox / "configs", "smoke"), sandbox)
    fp = fingerprint(ctx, STAGES["split"])
    with open(ctx.holdout, "a") as f:
        f.write("\n")
    assert fingerprint(ctx, STAGES["split"]) != fp
