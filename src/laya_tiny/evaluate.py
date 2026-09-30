"""Stage `eval`: score every backend on the human holdout with the same metrics.

Backends: the student (fp32 and int8 ONNX), the Laya teacher (its holdout predictions come from
the `label` stage), a TF-IDF + logistic-regression baseline trained on the same teacher labels,
and any external prediction files listed under `eval.external`."""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import Decision, decisions
from .export import encode, ort_session
from .metrics import accuracy, agreement, ece, macro_f1, percentile, tv_distance
from .stages import Ctx
from .teacher import read_labels
from .tok import load_tokenizer
from .util import log, read_jsonl, write_json

Probs = dict[str, np.ndarray]


def gold_index(dec: Decision, value: Any) -> int:
    if isinstance(value, bool):
        value = "true" if value else "false"
    return dec.labels.index(str(value))


def time_per_row(fn: Callable[[str], Any], texts: list[str], runs: int) -> dict[str, float]:
    for t in texts[:5]:
        fn(t)                                  # warm-up
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(texts[i % len(texts)])
        times.append((time.perf_counter() - t0) * 1000)
    return {"p50": percentile(times, 50), "p95": percentile(times, 95), "runs": runs}


class Student:
    def __init__(self, onnx_path: Path, tok_path: Path, names: list[str], max_len: int) -> None:
        self.sess, self.names, self.max_len, self.tok_path = ort_session(onnx_path), names, max_len, tok_path
        self.tok = load_tokenizer(tok_path)
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=max_len)

    def predict(self, texts: list[str]) -> Probs:
        ids, mask = encode(self.tok_path, texts, self.max_len)
        return dict(zip(self.names, self.sess.run(self.names, {"input_ids": ids, "attention_mask": mask})))

    def one(self, text: str) -> list[np.ndarray]:
        ids = np.asarray([self.tok.encode(text).ids], dtype=np.int64)
        return self.sess.run(self.names, {"input_ids": ids, "attention_mask": np.ones_like(ids)})


class TfidfBaseline:
    """Per-decision logistic regression on word+char TF-IDF, fitted to the teacher's argmax."""

    def __init__(self, decs: list[Decision], seed: int) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion

        self.decs = decs
        self.vec = FeatureUnion([
            ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=50000,
                                  sublinear_tf=True)),
        ])
        self.clf = {d.name: LogisticRegression(max_iter=2000, C=4.0, random_state=seed) for d in decs}

    def fit(self, rows: list[dict[str, Any]]) -> "TfidfBaseline":
        x = self.vec.fit_transform([r["text"] for r in rows])
        for d in self.decs:
            self.clf[d.name].fit(x, [int(np.argmax(r["probs"][d.name])) for r in rows])
        return self

    def predict(self, texts: list[str]) -> Probs:
        x = self.vec.transform(texts)
        out = {}
        for d in self.decs:
            clf = self.clf[d.name]
            p = np.zeros((len(texts), d.n))
            p[:, clf.classes_] = clf.predict_proba(x)
            out[d.name] = p
        return out


def score(decs: list[Decision], probs: Probs, gold: dict[str, np.ndarray], teacher: Probs | None,
          sel: np.ndarray) -> dict[str, Any]:
    res: dict[str, Any] = {"rows": int(sel.sum())}
    for d in decs:
        p, y = probs[d.name][sel], gold[d.name][sel]
        m = {"accuracy": accuracy(p, y), "macro_f1": macro_f1(p, y, d.n), "ece": ece(p, y)}
        if teacher is not None:
            m["teacher_agreement"] = agreement(p, teacher[d.name][sel])
            m["teacher_tv"] = tv_distance(p, teacher[d.name][sel])
        res[d.name] = m
    return res


def laya_latency(cfg: dict[str, Any], texts: list[str], rows: int) -> dict[str, float] | None:
    if rows <= 0 or cfg["task"]["teacher"]["kind"] != "laya":
        return None
    from .teacher import LayaTeacher

    tcfg = {**cfg, "task": {**cfg["task"], "teacher": {**cfg["task"]["teacher"], "device": "cpu"}}}
    teacher = LayaTeacher(tcfg)
    router = teacher._load()
    qs, model = teacher.questions, teacher.tcfg["checkpoint"]
    lat = time_per_row(lambda t: router.predict(t, qs, model=model), texts, rows)
    lat["what"] = "PyTorch CPU, batch=1, end-to-end"
    return lat


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def run(ctx: Ctx) -> dict[str, Any]:
    cfg, ecfg = ctx.cfg, ctx.cfg["eval"]
    decs = decisions(cfg)
    names = [d.name for d in decs]
    holdout = read_jsonl(ctx.holdout)
    texts = [h["text"] for h in holdout]
    gold = {d.name: np.asarray([gold_index(d, h[d.name]) for h in holdout]) for d in decs}
    main = (cfg["data"].get("languages") or ["en"])[0]
    lang_en = np.asarray([h.get("language", main) == main for h in holdout])      # the task's own language
    slices = {main: lang_en, "other_lang": ~lang_en}
    tlab = read_labels(ctx.path("label", "holdout.jsonl"))
    teacher = {d.name: np.asarray([tlab[h["id"]][d.name] for h in holdout]) for d in decs}
    teacher_kind = cfg["task"]["teacher"]["kind"]
    runs = int(ecfg["latency_runs"])
    en_texts = [t for t, e in zip(texts, lang_en) if e]
    exp = ctx.path("export")
    meta = json.loads((exp / "meta.json").read_text())
    backends: list[dict[str, Any]] = []

    def add(name: str, kind: str, probs: Probs, size: int, lat: dict[str, Any] | None, **extra: Any) -> None:
        entry = {"name": name, "kind": kind, "size_bytes": size, "latency_ms": lat, **extra,
                 "slices": {k: score(decs, probs, gold, teacher, s) for k, s in slices.items() if s.any()}}
        backends.append(entry)
        en = entry["slices"][main]
        log.info("%-22s %s  p50=%s ms", name, "  ".join(f"{n}={en[n]['accuracy']:.3f}" for n in names),
                 f"{lat['p50']:.2f}" if lat else "-")

    for fname, label in (("student.int8.onnx", "laya-tiny int8"), ("student.onnx", "laya-tiny fp32")):
        st = Student(exp / fname, exp / "tokenizer.json", names, int(meta["max_len"]))
        lat = time_per_row(st.one, en_texts, runs)
        lat["what"] = "onnxruntime CPU 1 thread, batch=1, incl. tokenisation"
        add(label, "student", st.predict(texts), (exp / fname).stat().st_size + (exp / "tokenizer.json").stat().st_size,
            lat, params=meta["params"])

    teacher_name = f"Laya {cfg['task']['teacher']['checkpoint']} fp32" if teacher_kind == "laya" else f"teacher ({teacher_kind})"
    size = 0
    if teacher_kind == "laya":
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == cfg["task"]["teacher"]["repo"]:
                size = sum(f.size_on_disk for rev in repo.revisions for f in rev.files
                           if f"/{cfg['task']['teacher']['checkpoint']}/" in str(f.file_path))
    add(teacher_name, "teacher", teacher, size, laya_latency(cfg, en_texts, int(ecfg["laya_latency_rows"])))

    train_rows = read_jsonl(ctx.path("split", "train.jsonl"))
    t0 = time.perf_counter()
    base = TfidfBaseline(decs, int(cfg["seed"])).fit(train_rows)
    log.info("tf-idf baseline fitted in %.1fs", time.perf_counter() - t0)
    blob = pickle.dumps(base)
    lat = time_per_row(lambda t: base.predict([t]), en_texts, runs)
    lat["what"] = "scikit-learn CPU, batch=1"
    add("TF-IDF + LogReg", "baseline", base.predict(texts), len(blob), lat)

    for ext in ecfg.get("external") or []:
        path = ctx.root / ext["predictions"]
        if not path.exists():
            log.warning("external predictions %s missing, skipped", path)
            continue
        pr = {r["id"]: r["probs"] for r in read_jsonl(path)}
        if not all(h["id"] in pr for h in holdout):
            log.warning("external predictions %s do not cover the holdout, skipped", path)
            continue
        probs = {d.name: np.asarray([pr[h["id"]][d.name] for h in holdout]) for d in decs}
        add(ext["name"], "external", probs, int(ext.get("size_bytes", 0)), ext.get("latency_ms"))

    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "profile": cfg["profile"],
           "main_slice": main, "task": cfg["task"].get("name"), "goal": cfg["task"].get("goal"),
           "holdout": {"path": cfg["holdout"], "rows": len(holdout), "en": int(lang_en.sum()),
                       "other_lang": int((~lang_en).sum()),
                       "source": ", ".join(sorted({h.get("source", "?") for h in holdout}))},
           "teacher": teacher_name, "decisions": {d.name: list(d.labels) for d in decs},
           "student": {k: meta[k] for k in ("params", "max_len", "temperatures", "run_id")},
           "backends": backends, "acceptance": acceptance(backends, names, cfg, main)}
    write_json(ctx.path("eval", "results.json"), res)
    return {"acceptance": {k: v["pass"] for k, v in res["acceptance"]["checks"].items()}}


def acceptance(backends: list[dict[str, Any]], names: list[str], cfg: dict[str, Any], main: str = "en") -> dict[str, Any]:
    acc = cfg["eval"].get("acceptance") or {}
    margin, max_mb, max_ms = float(acc.get("max_acc_drop", 0.03)), float(acc.get("max_int8_mb", 5)), float(acc.get("max_p50_ms", 5))
    student = next(b for b in backends if b["name"] == "laya-tiny int8")
    teacher = next(b for b in backends if b["kind"] == "teacher")
    checks: dict[str, Any] = {}
    for n in names:
        s, t = student["slices"][main][n]["accuracy"], teacher["slices"][main][n]["accuracy"]
        checks[f"accuracy/{n}"] = {"student": s, "teacher": t, "gap": s - t, "pass": s >= t - margin}
    mb = student["size_bytes"] / 1e6
    checks["size_int8_mb"] = {"value": mb, "limit": max_mb, "pass": mb <= max_mb}
    p50 = student["latency_ms"]["p50"]
    checks["cpu_p50_ms"] = {"value": p50, "limit": max_ms, "pass": p50 <= max_ms}
    return {"margin": margin, "checks": checks, "pass": all(c["pass"] for c in checks.values())}
