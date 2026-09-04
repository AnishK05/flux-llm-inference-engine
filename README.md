# flux-llm-inference-engine

Production-style LLM inference platform: serving, continuous batching, KV-cache management, streaming generation, and observability.

**Target setup:** Windows laptop, **CPU only**, `Qwen/Qwen2.5-0.5B-Instruct`. Develop in [WSL2](docs/windows-wsl2.md).

The full build plan is in [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md). This tree currently implements **Phase 0 (scaffold)** and **Phase 1 (naive baseline engine)**.

## Phase 0–1 status

| Piece | What it does |
|---|---|
| `make hello` | Allocate a 1000×1000 fp32 CPU tensor and print device / RSS / thread facts |
| `GET /health` | Same probe over HTTP (`model_loaded` is false until weights load) |
| `NaiveEngine` | Full-sequence `forward` every token, **no KV cache** — the baseline later phases beat |
| `POST /v1/completions` | OpenAI-shaped, non-streaming, one request at a time (`max_tokens` capped at 64) |

## Quickstart (WSL2 / Linux)

```bash
make install          # CPU PyTorch wheel + editable package
make hello            # Phase 0 probe
make test             # unit tests with FakeLM (no model download)
make api              # loads Qwen2.5-0.5B-Instruct (~1 GB first time)
```

In another terminal, after `/health` shows `"model_loaded": true`:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"flux-qwen-0.5b","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

Optional sidecars (not required for the API):

```bash
make compose-up    # redis, prometheus, grafana stubs
```

Grafana: http://localhost:3001 (anonymous viewer). Prometheus: http://localhost:9090.

## Config

See `.env.example`. Defaults are CPU / fp32 / Qwen 0.5B. Do not set `uvicorn --workers` above 1.

Integration test (downloads weights):

```bash
make test-integration
```
