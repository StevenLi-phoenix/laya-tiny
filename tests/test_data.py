import csv

import pytest

from laya_tiny.data import (build_corpus, clean_text, dedup_key, fill_placeholders, looks_english,
                            read_bitext, read_tobi_bueck, stable_bucket, truncate)

pytestmark = pytest.mark.unit


def test_clean_text_normalises_untrusted_input():
    raw = "Hello\\n\\nWorld\x07  <b>bold</b>​ \r\n\n\n\nend  "
    assert clean_text(raw) == "Hello\n\nWorld bold\n\nend"


def test_truncate_prefers_sentence_boundary():
    text = "First sentence here. Second sentence is long and keeps going on and on."
    assert truncate(text, 36) == "First sentence here."
    assert truncate(text, 60).endswith("long and keeps")   # no sentence end in the back half: cut at a word
    assert truncate("short", 40) == "short"
    assert len(truncate("word " * 50, 33)) <= 33


def test_dedup_key_and_language():
    assert dedup_key("Hello,  WORLD!") == dedup_key("hello world")
    assert looks_english("My invoice is wrong")
    assert not looks_english("我三月份被重复扣款了")
    assert not looks_english("12345 !!!")


def test_placeholders_are_filled_deterministically():
    a = fill_placeholders("cancel order {{Order Number}} for {{Unknown Thing}}", "seed")
    assert "{{" not in a and "unknown thing" in a
    assert a == fill_placeholders("cancel order {{Order Number}} for {{Unknown Thing}}", "seed")


def _csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_source_readers_map_weak_labels(tmp_path):
    tb = tmp_path / "tb.csv"
    _csv(tb, ["subject", "body", "queue", "priority", "language"], [
        ["Refund", "Please refund\\nme", "Billing and Payments", "high", "en"],
        ["", "Hallo", "IT Support", "low", "de"],
        ["nan", "Other body", "Customer Service", "low", "en"],
    ])
    rows = list(read_tobi_bueck(tb, {"Billing and Payments": {"department": "billing"}}, ["en"]))
    assert [r["text"] for r in rows] == ["Refund\n\nPlease refund\\nme", "Other body"]
    assert rows[0]["weak"] == {"department": "billing"} and rows[1]["weak"] == {}
    bx = tmp_path / "bx.csv"
    _csv(bx, ["flags", "instruction", "category", "intent", "response"],
         [["B", "where is my invoice {{Invoice Number}}", "INVOICE", "get_invoice", "..."]])
    (r,) = read_bitext(bx, {"get_invoice": {"department": "billing"}})
    assert "{{" not in r["text"] and r["weak"] == {"department": "billing"}


def test_build_corpus_dedups_and_caps(smoke_cfg, tmp_path):
    cfg = {**smoke_cfg, "data": {**smoke_cfg["data"], "max_rows": 50}}
    rows, summary = build_corpus(cfg, tmp_path)
    assert len(rows) == 50 == summary["rows"]
    assert len({r["id"] for r in rows}) == 50
    assert len({dedup_key(r["text"]) for r in rows}) == 50
    again, _ = build_corpus(cfg, tmp_path)
    assert again == rows                                  # deterministic


def test_stable_bucket_range():
    vals = [stable_bucket(str(i)) for i in range(1000)]
    assert all(0 <= v < 1 for v in vals) and 0.4 < sum(vals) / len(vals) < 0.6
