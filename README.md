# flux-llm-inference-engine

Production-style LLM inference platform: serving, continuous batching, KV-cache management, streaming generation, and observability.

**Target setup:** Windows laptop, **CPU only**, `Qwen/Qwen2.5-0.5B-Instruct`. Develop in [WSL2](docs/windows-wsl2.md).

The full build plan is in [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md). This tree implements **Phases 0–11**.

The engine is device-agnostic; this repo is validated on CPU.

## Status

| Piece | What it does |
|---|---|
| `make hello` | Allocate a 1000×1000 fp32 CPU tensor and print device / RSS / thread facts |
| `GET /health` | Probe + `serve_engine` (`continuous` by default) |
| `GET /admin/stats` | Waiting / running ids, last batch size, KV, RSS, tok/s, p50 TTFT |
| `GET /metrics` | Prometheus text |
| Next.js dashboard | Playground, live engine (the interview page), bench table |
| Compose | `api` + `dashboard` + redis + prometheus + grafana |

## One-command demo (WSL2 + Docker Desktop)

```bash
docker compose up --build
```

First run downloads **~1 GB** of Qwen2.5-0.5B-Instruct into the `hf-cache` volume. Then:

| URL | What |
|---|---|
| http://127.0.0.1:3000 | Playground |
| http://127.0.0.1:3000/live | Live engine (batch, KV, queue, RSS) |
| http://127.0.0.1:3000/bench | Last `naive_vs_flux` table |
| http://127.0.0.1:8000/docs | FastAPI |
| http://127.0.0.1:3001 | Grafana (admin / `flux`) |

Compose sets `FLUX_ENABLE_REDIS=true` for IP rate limits and job status (`GET /v1/requests/{id}`). Local `make api` keeps Redis off.

## Local quickstart (no Docker)

```bash
make install
make test
make api
```

In another terminal:

```bash
make dashboard    # Next.js on :3000 → API on :8000
```

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"flux-qwen-0.5b","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16,"temperature":0,"stream":true}'
```

## Benchmarks (Phase 8)

```bash
make bench          # Qwen, writes docs/benchmark_results.md + SVG plots
make bench-quick    # FakeLM, CI-safe
```

Re-run `make bench` on the Windows + WSL2 laptop (plugged in) for resume numbers. Do not quote `soak_200` e2e p99 as a latency win.

![p99 TTFT](docs/bench_ttft_p99.svg)

![throughput](docs/bench_tok_s.svg)

## Observability (Phase 9)

Grafana: http://127.0.0.1:3001 — dashboard **Flux serving**. Password `flux`.

![Grafana Flux serving](docs/grafana-serving.webp)

Serving modes: `continuous` (default), `queued`, `cached`, `naive`. Do not set `uvicorn --workers` above 1.

```bash
make test-integration
```
