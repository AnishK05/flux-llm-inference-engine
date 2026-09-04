# flux-llm-inference-engine

Flux is a lightweight **LLM inference engine** that serves causal language models
over an **OpenAI-compatible HTTP API**, with a built-in streaming chat UI.

It implements a real token-by-token decode loop — batched generation with a
reused KV cache, left-padding with explicit position ids, and per-request
sampling (temperature, top-k, top-p, repetition penalty) — rather than wrapping a
black-box `generate()` call. It runs on CPU out of the box using a small default
model (`distilgpt2`).

## Features

- **Batched engine** (`flux/engine.py`): a single worker thread groups pending
  requests into a static batch and decodes them in lockstep, reusing a batched
  KV cache. Left-padding + explicit `position_ids` keep mixed-length prompts
  correct.
- **Streaming**: tokens are pushed to each caller through a per-request queue and
  forwarded as Server-Sent Events.
- **OpenAI-compatible API**: `/v1/completions`, `/v1/chat/completions` (both with
  `stream: true`), `/v1/models`, and `/health`.
- **Web chat UI** at `/` with live token streaming.
- **Configurable** entirely via environment variables (`FLUX_MODEL`,
  `FLUX_MAX_BATCH_SIZE`, `FLUX_PORT`, …).

## Quick start

```bash
bash scripts/setup.sh        # create venv, install deps, cache the model
bash scripts/start.sh        # serve on http://localhost:8000
```

Then open <http://localhost:8000> for the chat UI, or call the API:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":32,"stream":true}'
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `FLUX_MODEL` | `distilgpt2` | Any Hugging Face causal LM id. |
| `FLUX_DEVICE` | `cpu` | Torch device (`cpu`, `cuda`). |
| `FLUX_MAX_BATCH_SIZE` | `4` | Requests decoded together per batch. |
| `FLUX_MAX_NEW_TOKENS` | `512` | Hard cap on generated tokens per request. |
| `FLUX_MAX_PROMPT_TOKENS` | `1024` | Reject prompts longer than this. |
| `FLUX_HOST` / `FLUX_PORT` | `0.0.0.0` / `8000` | Server bind address. |

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest           # unit + engine + API tests
```

## Project layout

```
flux/
  config.py      # env-driven configuration
  sampling.py    # temperature / top-k / top-p / repetition penalty
  engine.py      # batched, streaming decode loop with KV cache
  protocol.py    # OpenAI-compatible pydantic models
  server.py      # FastAPI app + endpoints
  web/index.html # streaming chat UI
scripts/
  setup.sh       # idempotent environment bootstrap
  start.sh       # run the server
tests/           # pytest suite
```
