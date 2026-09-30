import pytest

from laya_tiny.tok import CLS_ID, PAD_ID, UNK_ID, coverage, load_tokenizer, train_tokenizer

pytestmark = pytest.mark.unit

TEXTS = ["I was charged twice for my invoice, please refund.", "The app crashes when I open settings.",
         "Can't log in, password reset email never arrives.", "How much is the enterprise plan for 50 seats?"] * 30


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(TEXTS, vocab_size=400, min_frequency=1, max_len=32)


def test_specials_and_cls(tok):
    enc = tok.encode("refund please")
    assert enc.ids[0] == CLS_ID and PAD_ID not in enc.ids


@pytest.mark.parametrize("text", ["I was charged twice.", "Émile's café — naïve ☕", "我三月份被重复扣款了", "  spaced  out "])
def test_roundtrip_without_unk(tok, text):
    tok.no_truncation()
    enc = tok.encode(text)
    assert UNK_ID not in enc.ids
    assert tok.decode(enc.ids, skip_special_tokens=True).strip() == text.strip()
    tok.enable_truncation(max_length=32)


def test_truncation_padding_and_save(tok, tmp_path):
    batch = tok.encode_batch(["word " * 100, "short"])
    assert len(batch[0].ids) == len(batch[1].ids) == 32
    assert batch[1].attention_mask.count(0) > 0
    p = tmp_path / "tokenizer.json"
    tok.save(str(p))
    assert load_tokenizer(p).encode("refund").ids == tok.encode("refund").ids


def test_coverage_stats(tok):
    st = coverage(tok, TEXTS[:4], max_len=32)
    assert st["unk_rate"] == 0 and st["mean_tokens"] > 3 and st["truncated_frac"] == 0
