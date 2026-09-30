"""The holdout is the only human-labelled data; guard its schema and balance."""
import hashlib
from collections import Counter

import pytest

from laya_tiny.config import decisions, load_config
from laya_tiny.util import read_jsonl

pytestmark = pytest.mark.unit


def test_holdout_schema_and_size(repo_root):
    cfg = load_config(repo_root / "configs")
    rows = read_jsonl(repo_root / cfg["holdout"])
    decs = {d.name: d for d in decisions(cfg)}
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["id"] == "h-" + hashlib.sha1(r["text"].encode()).hexdigest()[:12]
        assert r["department"] in decs["department"].labels
        assert str(r["urgency"]) in decs["urgency"].labels
        assert isinstance(r["churn_risk"], bool)
        assert r["language"] and r["source"]
    en = [r for r in rows if r["language"] == "en"]
    assert len(en) >= 200
    dept = Counter(r["department"] for r in en)
    assert min(dept.values()) >= 40
    assert sum(r["churn_risk"] for r in en) >= 30
