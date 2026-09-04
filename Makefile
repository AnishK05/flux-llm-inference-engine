.PHONY: install test test-integration api hello bench bench-kv compose-up compose-down

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
TORCH_CPU_INDEX := https://download.pytorch.org/whl/cpu
export HF_HOME ?= $(CURDIR)/.hf-cache

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip wheel

install: $(VENV)/bin/python
	$(PIP) install torch==2.6.0 --index-url $(TORCH_CPU_INDEX)
	$(PIP) install -e ".[dev]" --extra-index-url $(TORCH_CPU_INDEX)

test: $(VENV)/bin/python
	$(BIN)/pytest -q

test-integration: $(VENV)/bin/python
	FLUX_RUN_INTEGRATION=1 $(BIN)/pytest -q -m integration

hello: $(VENV)/bin/python
	$(PY) scripts/hello_cpu.py

api: $(VENV)/bin/python
	$(BIN)/uvicorn flux.server.app:app --host 0.0.0.0 --port 8000

bench:
	@echo "Full loadgen lands in Phase 8. For now: make bench-kv"

bench-kv: $(VENV)/bin/python
	$(PY) benchmarks/compare_naive_vs_cached.py --qwen --lengths 32,128 --max-tokens 4 --out docs/phase2_naive_vs_cached.json

compose-up:
	docker compose up -d redis prometheus grafana

compose-down:
	docker compose down
