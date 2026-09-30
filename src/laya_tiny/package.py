"""Stage `package`: the deliverable. A self-contained folder with the int8 model, tokenizer,
label contract, a dependency-light `predict.py` and a README with the evaluation numbers."""
from __future__ import annotations

import json
import shutil
from typing import Any

from .config import decisions
from .stages import Ctx
from .util import read_json, write_json

PREDICT_PY = '''#!/usr/bin/env python3
"""Run the {name} expert model. Needs only: pip install onnxruntime tokenizers numpy

  python predict.py "some text"                 # one text -> JSON
  python predict.py < texts.txt                 # one text per line -> JSON lines
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
META = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
TOK = Tokenizer.from_file(str(HERE / "tokenizer.json"))
TOK.no_padding()
TOK.enable_truncation(max_length=META["max_len"])
SESS = ort.InferenceSession(str(HERE / "model.onnx"), providers=["CPUExecutionProvider"])


def predict(text):
    ids = np.asarray([TOK.encode(text).ids], dtype=np.int64)
    outs = SESS.run(META["outputs"], {{"input_ids": ids, "attention_mask": np.ones_like(ids)}})
    result = {{}}
    for name, probs in zip(META["outputs"], outs):
        p = probs[0].tolist()
        labels = META["labels"][name]
        best = max(range(len(p)), key=p.__getitem__)
        result[name] = {{"label": labels[best], "confidence": round(p[best], 4),
                        "probabilities": {{l: round(x, 4) for l, x in zip(labels, p)}}}}
    return result


if __name__ == "__main__":
    texts = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [l.rstrip("\\n") for l in sys.stdin if l.strip()]
    for t in texts:
        print(json.dumps({{"text": t, **predict(t)}}, ensure_ascii=False))
'''


def _fmt(x: Any) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f}%"


def run(ctx: Ctx) -> dict[str, Any]:
    cfg = ctx.cfg
    exp, out = ctx.path("export"), ctx.path("package")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(exp / "student.int8.onnx", out / "model.onnx")
    shutil.copy2(exp / "tokenizer.json", out / "tokenizer.json")
    meta = read_json(exp / "meta.json")
    res = read_json(ctx.path("eval", "results.json"))
    task = cfg["task"]
    name = task.get("name", "model")
    decs = decisions(cfg)
    student = next(b for b in res["backends"] if b["name"] == "laya-tiny int8")
    teacher = next(b for b in res["backends"] if b["kind"] == "teacher")
    main = res.get("main_slice", "en")
    model_meta = {
        "name": name, "goal": task.get("goal"), "summary": task.get("summary"), "outputs": meta["outputs"],
        "labels": meta["labels"], "max_len": meta["max_len"], "temperatures": meta["temperatures"],
        "questions": {d.name: d.question for d in decs}, "params": meta["params"],
        "teacher": meta.get("teacher"), "files": meta["files"],
        "eval": {"holdout": res["holdout"], "student": student["slices"].get(main),
                 "teacher": teacher["slices"].get(main), "latency_ms": student["latency_ms"]},
    }
    write_json(out / "model.json", model_meta)
    (out / "predict.py").write_text(PREDICT_PY.format(name=name), encoding="utf-8")
    rows = []
    for d in decs:
        s, t = (student["slices"].get(main) or {}).get(d.name, {}), (teacher["slices"].get(main) or {}).get(d.name, {})
        labels = ", ".join(d.labels) if d.type != "score" else " < ".join(str(c).split(":")[0] for c in d.question["criteria"])
        rows.append(f"| `{d.name}` | {d.type} | {labels} | {_fmt(s.get('accuracy'))} | {_fmt(t.get('accuracy'))} | "
                    f"{_fmt(s.get('teacher_agreement'))} |")
    size_mb = (out / "model.onnx").stat().st_size / 1e6
    readme = f"""# {name}

{task.get('summary') or ''}

Goal: *{task.get('goal') or '(hand-written task)'}*

A {meta['params'] / 1e6:.1f}M-parameter Transformer distilled from Laya `{(meta.get('teacher') or {}).get('checkpoint', '?')}`
by [laya-tiny](https://github.com/StevenLi-phoenix/laya-tiny). `model.onnx` is {size_mb:.1f} MB (int8);
CPU latency p50 {student['latency_ms']['p50']:.2f} ms per text.

```bash
pip install onnxruntime tokenizers numpy
python predict.py "your text here"
```

| decision | type | labels | accuracy (student) | accuracy (teacher) | agrees with teacher |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Evaluated on {res['holdout']['rows']} held-out rows ({res['holdout'].get('source', 'see report')}). For a goal-driven
project the holdout is also written by the LLM (labels = what it was asked to write), so read these numbers as
"agrees with the task definition", not as human-verified accuracy.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    json.dumps(model_meta)  # fail loudly if anything non-serialisable slipped in
    return {"package": str(out), "model_mb": round(size_mb, 2)}
