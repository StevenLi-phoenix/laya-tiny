"""Stage `tokenizer`: train a byte-level BPE on the *training split only* and report coverage."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

from .stages import Ctx
from .util import iter_jsonl, log, read_jsonl, write_json

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3


def train_tokenizer(texts: Iterable[str], vocab_size: int, min_frequency: int = 2,
                    lowercase: bool = False, max_len: int = 192) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    norms: list[Any] = [normalizers.NFC()]
    if lowercase:
        norms.append(normalizers.Lowercase())
    tok.normalizer = normalizers.Sequence(norms)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    assert [tok.token_to_id(s) for s in SPECIALS] == [PAD_ID, UNK_ID, CLS_ID, SEP_ID]
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A", special_tokens=[("[CLS]", CLS_ID)])
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(pad_id=PAD_ID, pad_token="[PAD]")
    return tok


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def coverage(tok: Tokenizer, texts: list[str], max_len: int) -> dict[str, float]:
    if not texts:
        return {"texts": 0}
    tok.no_truncation()
    tok.no_padding()
    lens, unk, total, chars = [], 0, 0, 0
    for enc in tok.encode_batch(texts):
        lens.append(len(enc.ids))
        unk += sum(1 for i in enc.ids if i == UNK_ID)
        total += len(enc.ids)
    chars = sum(len(t) for t in texts)
    lens.sort()
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(pad_id=PAD_ID, pad_token="[PAD]")
    return {"texts": len(texts), "unk_rate": unk / max(total, 1), "mean_tokens": statistics.fmean(lens),
            "p95_tokens": float(lens[int(0.95 * (len(lens) - 1))]), "chars_per_token": chars / max(total, 1),
            "truncated_frac": sum(1 for n in lens if n > max_len) / len(lens)}


def run(ctx: Ctx) -> dict[str, Any]:
    mcfg = ctx.cfg["model"]
    tcfg, max_len = mcfg["tokenizer"], int(mcfg["encoder"]["max_len"])
    tok = train_tokenizer((r["text"] for r in iter_jsonl(ctx.path("split", "train.jsonl"))),
                          int(tcfg["vocab_size"]), int(tcfg.get("min_frequency", 2)),
                          bool(tcfg.get("lowercase", False)), max_len)
    out = ctx.path("tokenizer", "tokenizer.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    val = [r["text"] for r in read_jsonl(ctx.path("split", "val.jsonl"))]
    holdout = read_jsonl(ctx.holdout)
    main = (ctx.cfg["data"].get("languages") or ["en"])[0]
    stats: dict[str, Any] = {"vocab_size": tok.get_vocab_size(), "val": coverage(tok, val, max_len)}
    for key, rows in ((f"holdout_{main}", [h["text"] for h in holdout if h.get("language", main) == main]),
                      ("holdout_other_lang", [h["text"] for h in holdout if h.get("language", main) != main])):
        if rows:
            stats[key] = coverage(tok, rows, max_len)
    write_json(ctx.path("tokenizer", "stats.json"), stats)
    for k, v in stats.items():
        log.info("tokenizer %s: %s", k, v)
    return {"vocab_size": stats["vocab_size"], "val_mean_tokens": round(stats["val"]["mean_tokens"], 1),
            "val_truncated": round(stats["val"]["truncated_frac"], 4)}
