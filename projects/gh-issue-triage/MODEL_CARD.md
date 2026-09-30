# gh-issue-triage

Classify GitHub issues by type, severity, and reproducibility.

Goal: *Triage GitHub issues for an open-source library: what kind of issue it is, how severe it is, and whether it includes enough information to reproduce*

A 3.1M-parameter Transformer distilled from Laya `typed-decisions`
by [laya-tiny](https://github.com/StevenLi-phoenix/laya-tiny). `model.onnx` is 3.2 MB (int8);
CPU latency p50 0.63 ms per text.

```bash
pip install onnxruntime tokenizers numpy
python predict.py "your text here"
```

| decision | type | labels | accuracy (student) | accuracy (teacher) | agrees with teacher |
|---|---|---|---|---|---|
| `issue_type` | choice | bug, feature, docs, question, other | 56.3% | 52.8% | 77.0% |
| `severity` | score | low < medium < high < critical | 59.1% | 57.1% | 61.9% |
| `reproducible` | noul | false, true | 78.6% | 68.3% | 73.8% |

Evaluated on 252 held-out rows (llm-synth-holdout). For a goal-driven
project the holdout is also written by the LLM (labels = what it was asked to write), so read these numbers as
"agrees with the task definition", not as human-verified accuracy.
