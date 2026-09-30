# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Synthetic-data requests whose JSON stayed broken after 3 retries (typically texts containing code with
  unescaped quotes) were dropped. They now fall back to one plain-text request asking for one text per line;
  recovered rows are tagged `meta.fallback = "lines"`, and only requests that still yield nothing are skipped.

## [0.2.0] - 2026-09-30

### Added
- `laya-tiny new "<goal>"` / `make new GOAL=...`: one goal in, a packaged expert model out. An LLM designs the
  decision contract (validated, retried with feedback) and writes stratified synthetic training texts and a
  synthetic holdout; Laya labels them; the existing pipeline trains, exports and evaluates.
- `llm.py`: OpenAI-compatible client that stores every raw response on disk before parsing (request-hash cache),
  treats output as untrusted JSON, and never logs the API key. `fake_llm.py` for offline tests.
- `package` stage: `model.onnx` (int8) + `tokenizer.json` + `model.json` + standalone `predict.py` + README.
- `--project <name>` for every command; project `task.yaml` replaces the default task, `project.yaml` overrides.
- `smoke-new` profile and a CI step running the goal-driven flow offline.

### Changed
- Evaluation, report and tokenizer stats use the task's language as the main slice instead of hard-coded English.
- `package` is now the final pipeline stage (`make smoke` runs through it).

### Fixed
- `dedup_key` kept only ASCII letters, so every non-Latin text collapsed to the same key and was dropped as a
  duplicate.

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
- Teacher pinned to fp32 (`task.teacher.precision`); Laya otherwise autocasts to fp16 on MPS / bf16 on CUDA.
- First full run: 30k rows labelled on an RTX 4070 Ti SUPER; report in `reports/2026-09-29.md`. Size and latency
  targets met (4.0 MB, 0.51 ms); accuracy targets not met (−19.5 / −3.9 / −6.3 pp), diagnosed as distribution shift.
- `smoke` profile (templated corpus + keyword teacher, CPU, ~10 s) for tests and CI.
- GitHub Actions: pytest + smoke on every push/PR; a manual/monthly full run with the Laya teacher on CPU.
