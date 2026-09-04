# flux-llm-inference-engine

Production-style LLM inference platform: serving, continuous batching, KV-cache management, streaming generation, and observability.

**Target setup:** Windows laptop, **CPU only**, `Qwen/Qwen2.5-0.5B-Instruct`. Develop in [WSL2](docs/windows-wsl2.md).

The full build plan is in [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md). This tree implements **Phases 0–3**.

## Status

| Piece | What it does |
|---|---|
| `make hello` | Allocate a 1000×1000 fp32 CPU tensor and print device / RSS / thread facts |
| `GET /health` | Probe + `serve_engine` (`cached` by default) |
| `NaiveEngine` | Full-sequence `forward` every token, **no KV cache** — kept as the baseline |
| `CachedEngine` | Prefill once, then decode `[batch, 1]` with a growing KV cache |
| `POST /v1/completions` | OpenAI-shaped prompt completions (non-streaming, `max_tokens` cap 64) |
| `POST /v1/chat/completions` | Chat template + greedy/temperature/top-p sampling |

## Quickstart (WSL2 / Linux)

```bash
make install          # CPU PyTorch wheel + editable package
make hello            # Phase 0 probe
make test             # unit tests with FakeLM (no model download)
make api              # loads Qwen2.5-0.5B-Instruct (~1 GB first time)
```

After `/health` shows `"model_loaded": true`:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

curl -s http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"flux-qwen-0.5b","prompt":"The capital of France is","max_tokens":8,"temperature":0}'

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"flux-qwen-0.5b","messages":[{"role":"system","content":"You are concise."},{"role":"user","content":"What is a KV cache in one sentence?"}],"max_tokens":32,"temperature":0}'
```

Set `FLUX_SERVE_ENGINE=naive` to serve the Phase 1 baseline instead of the cached engine.

Naive vs cached microbench (downloads Qwen if needed):

```bash
make bench-kv
```

Optional sidecars: `make compose-up` (redis, prometheus, grafana stubs).

## Config

See `.env.example`. Defaults are CPU / fp32 / Qwen 0.5B / cached engine. Do not set `uvicorn --workers` above 1.

```bash
make test-integration   # downloads weights; greedy equality + KV shapes
```
