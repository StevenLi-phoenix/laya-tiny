# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-29

### Added
- Hash-cached stage pipeline (`data → label → split → tokenizer → train → calibrate → export → eval → report`),
  driven by `make <stage>` or `laya-tiny run <stage>`; a stage re-runs only when its code, its config subset or
  an input file changes.
- Fixed decision contract in `configs/task.yaml` (department ×4, urgency ×3, churn_risk ×2) with the exact Laya
  questions, pinned teacher checkpoint (`typed-decisions`) and Hub revision.
- Data stage: downloads two public support-ticket corpora, cleans untrusted text, filters by language and length,
  truncates at sentence boundaries, de-duplicates, and derives weak department labels from source metadata.
- Resumable teacher labelling with a per-teacher JSONL cache (torn writes tolerated); full probability
  distributions are stored with the checkpoint, revision, temperatures and questions they came from.
- Leakage guard: corpus rows whose word-3-gram Jaccard with any holdout row is ≥ 0.6 are dropped before splitting.
- Byte-level BPE tokenizer (8k) trained on the training split only; coverage stats for val and holdout.
- Student: 4-layer pre-LN Transformer encoder (d=192, 4 heads, max_len 192, ~3.3M params) with one head per decision;
  KL distillation at T=2 plus weak-label cross-entropy; AdamW, warmup + cosine, early stopping on teacher agreement.
- Per-head temperature calibration on the validation split.
- ONNX export (opset 17, dynamic batch/sequence) verified against PyTorch (≤ 1e-4), plus dynamic int8 quantisation
  including the embedding table.
- Evaluation on a 224-row human-labelled holdout (205 English): accuracy, macro-F1, ECE, teacher agreement and TV
  distance per decision; CPU latency and size for the student, the Laya teacher, a TF-IDF + logistic-regression
  baseline and external predictions (e.g. Laya q4 via `tools/laya_onnx_predict.mjs`).
- Markdown/JSON reports with an overview chart and acceptance checks.
- `smoke` profile (templated corpus + keyword teacher, CPU, ~10 s) for tests and CI.
- GitHub Actions: pytest + smoke on every push/PR; a manual/monthly full run with the Laya teacher on CPU.
