.PHONY: install test test-integration api hello bench bench-kv bench-batch bench-quick dashboard compose-up compose-down

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

bench: $(VENV)/bin/python
	$(PY) -m benchmarks.run_phase8 --qwen --scenarios naive_vs_flux,soak_200 --trials 3 --warmup 10 --requests 16 --out-dir docs

bench-quick: $(VENV)/bin/python
	$(PY) -m benchmarks.run_phase8 --scenarios naive_vs_flux,soak_200 --trials 1 --warmup 2 --requests 8 --out-dir docs

bench-kv: $(VENV)/bin/python
	$(PY) benchmarks/compare_naive_vs_cached.py --qwen --lengths 32,128 --max-tokens 4 --out docs/phase2_naive_vs_cached.json

bench-batch: $(VENV)/bin/python
	$(PY) benchmarks/queued_vs_continuous.py --qwen --concurrency 4,8 --max-tokens 8 --out docs/phase5_queued_vs_continuous.json

# Prefer Compose V2 (`docker compose`); fall back to the v1 binary.
COMPOSE ?= $(shell docker compose version >/dev/null 2>&1 && echo "docker compose" || echo docker-compose)

dashboard:
	cd dashboard && npm install && npm run dev

compose-up:
	$(COMPOSE) up -d --build

compose-down:
	$(COMPOSE) down
