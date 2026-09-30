# laya-tiny pipeline. Every stage target runs that stage plus anything stale upstream;
# freshness is decided by input hashes (see src/laya_tiny/stages.py), not file mtimes.
#   make new GOAL="triage GitHub issues by type and severity"   one goal in -> projects/<name>/work/package
#   make report              full pipeline on the hand-written ticket task (default profile, Laya teacher)
#   make smoke               offline CPU run in ~10 s (keyword teacher, templated corpus)
#   make train FORCE=train   re-run one stage even if fresh
#   make report SET="data.max_rows=4000 eval.laya_latency_rows=0"
PY      ?= .venv/bin/python
PROFILE ?=
FORCE   ?=
SET     ?=
LAYA_TS ?= ../laya-webgpu/vendor/laya-ts
LAYA_Q4 ?= ../laya-webgpu/model
GOAL    ?=
NAME    ?=
ROWS    ?= 3000
CLI      = $(PY) -m laya_tiny.cli $(if $(PROFILE),--profile $(PROFILE)) $(if $(PROJECT),--project $(PROJECT)) $(foreach s,$(SET),--set $(s))
STAGES   = data label split tokenizer train calibrate export eval report package

.PHONY: setup test smoke smoke-new new status laya-q4 clean-smoke $(STAGES)

setup:
	uv venv -p 3.12 .venv
	uv pip install --python $(PY) -e ".[dev,teacher]"

$(STAGES):
	$(CLI) run $@ $(if $(FORCE),--force $(FORCE))

status:
	$(CLI) status

test:
	$(PY) -m pytest -q

smoke:
	$(MAKE) package PROFILE=smoke

# Goal-driven: the LLM (llm.base_url, default a local llama-server) designs the task and writes the
# data, Laya labels it, and a packaged expert model lands in projects/<name>/work/package.
new:
	@test -n "$(GOAL)" || (echo 'usage: make new GOAL="what the model should decide" [NAME=my-model] [ROWS=3000]'; exit 2)
	$(PY) -m laya_tiny.cli $(if $(PROFILE),--profile $(PROFILE)) $(foreach s,$(SET),--set $(s)) new "$(GOAL)" \
	  $(if $(NAME),--name $(NAME)) --rows $(ROWS)

smoke-new:
	$(MAKE) new PROFILE=smoke-new GOAL="triage support tickets" NAME=smoke-demo ROWS=300

# Laya q4 (split ONNX, laya-ts on CPU) on the holdout -> work/external/laya-q4.jsonl.
# Needs a laya-webgpu checkout with `pnpm install` done and the q4 model downloaded.
laya-q4:
	mkdir -p work/external
	node tools/laya_onnx_predict.mjs $(LAYA_TS) $(LAYA_Q4) data/holdout/tickets.jsonl work/external/laya-q4.jsonl \
	  | tee work/external/laya-q4.summary.json

clean-smoke:
	rm -rf work-smoke
