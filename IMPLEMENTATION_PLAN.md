# Flux — LLM Inference Infrastructure

## Implementation Plan

This document is the build plan for **Flux**, a production-style LLM inference platform. The goal is not to train a model or invent a new architecture. The goal is to take an already-trained transformer and turn it into a **reliable serving system**: load it, queue requests, batch work, manage KV cache, stream tokens, measure latency vs. throughput, and explain the tradeoffs.

This is an undergrad SWE learning project. The scope is **high-value and interview-ready**, not a vLLM clone and not a multi-cluster production deployment.

---

## 0. How to use this document

- Build in the **phase order**. Later phases assume earlier ones exist and have a baseline to compare against.
- Treat the **naive baseline** as a first-class artifact. Resume numbers only mean something if you can show *before vs. after* on the same hardware and the same model.
- Every phase has: goal, what you build, what you do **not** build, learning checkpoint, acceptance criteria, and suggested tests.
- Open questions for you are collected in [Section 18](#18-open-questions-for-you) and also flagged inline as `QUESTION`.
- When a decision is still open, this plan recommends a **default** so you can start without waiting.

**Default project name:** Flux  
**Default API style:** OpenAI-compatible *subset* (`/v1/chat/completions` + `/v1/completions`) plus internal admin/metrics endpoints.  
**Default model (dev / CPU):** `Qwen2.5-0.5B-Instruct` or `TinyLlama-1.1B-Chat`.  
**Default model (GPU, if you have ≥8GB VRAM):** `Qwen2.5-1.5B-Instruct` or `Llama-3.2-1B-Instruct`.  
**Default concurrency target:** 200 in-flight requests against a **small** model with continuous batching — not 70B.

---

## 1. Project thesis (what you are actually learning)

Modern AI products are usually bottlenecked by **inference**, not training. A trained model sitting in a `.safetensors` file does not serve users. Serving it well requires answering:

1. How do I generate the next token without recomputing the entire prompt every time? (**KV cache**)
2. How do I keep the GPU busy when requests arrive at different times and finish at different lengths? (**continuous / iteration-level batching**)
3. How do I admit work when KV memory is the real scarce resource, not CPU threads? (**scheduler + admission control**)
4. How do I show the user tokens as they are produced? (**streaming**)
5. How do I know whether a change helped? (**benchmarks + Prometheus/Grafana**)

If you can explain those five with diagrams, code you wrote, and plots from your own harness, this project has done its job.

### Resume line this project is aiming at

> Built a production-style LLM inference server in Python and FastAPI with continuous batching, KV-cache reuse, and request queuing, sustaining 200+ concurrent requests while cutting p99 latency 45% and lifting throughput 3x.

Those numbers are **directionally right** and should be **measured, not invented**. After the benchmark harness exists, replace them with whatever your machine actually produces. A true 2.1x throughput win with a plot is stronger than a fake 3x.

---

## 2. What this project is / is not

### It is

- A **from-scratch serving stack** around Hugging Face weights (you own the batching, cache, scheduler, and HTTP layer).
- A **measurable** comparison: naive sequential serving vs. KV-cached decode vs. continuous batching.
- A **small but real** control plane: queue, backpressure, streaming, metrics, a playground UI.
- A system you can talk about in interviews for 20+ minutes without hand-waving.

### It is not

- Training, fine-tuning, RLHF, or prompt-engineering as the main work.
- A from-scratch GPT trained on Tiny Shakespeare as the product (a tiny educational decoder is optional in Phase 1 only).
- A reimplementation of vLLM, TensorRT-LLM, TGI, or Triton.
- Multi-node tensor/pipeline parallelism, Kubernetes autoscaling, or multi-tenant billing.
- Custom CUDA kernels. Using CUDA via PyTorch is enough; writing FlashAttention is out of scope.
- Full OpenAI API compatibility (tools, vision, assistants, batch API, etc.).

---

## 3. Target architecture

Keep a **single-process inference engine** as the hot path. FastAPI accepts HTTP, the engine owns the GPU/CPU, Redis is for control-plane state (optional on the hot path), and the dashboard is a client of the API + metrics.

```text
                         ┌─────────────────────────────────────┐
                         │           Next.js dashboard         │
                         │  playground · queue · live metrics  │
                         └──────────────────┬──────────────────┘
                                            │ HTTP / SSE
┌─────────────┐          ┌──────────────────▼──────────────────┐
│ Prometheus  │ scrape   │             FastAPI gateway         │
│ Grafana     │◀─────────│  /v1/chat/completions  (stream)     │
└─────────────┘          │  /v1/completions                    │
                         │  /health  /metrics  /admin/stats    │
                         └──────────────────┬──────────────────┘
                                            │ in-process
                         ┌──────────────────▼──────────────────┐
                         │            Engine (asyncio)         │
                         │  RequestQueue → Scheduler → Worker  │
                         │  Prefill / Decode loop              │
                         │  KV Block Pool · Tokenizer          │
                         │  Sampler (temp, top_p, max_tokens)  │
                         └──────────────────┬──────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
              PyTorch model            Redis (optional)        Hugging Face
              GPU or CPU               job status, rate        weights + tok
                                       limits, prefix keys
```

### Process model (recommended)

| Component | Process | Why |
|---|---|---|
| Inference engine + FastAPI | **One Python process**, asyncio, one model replica | Avoids N copies of weights. The GPU is the bottleneck, not FastAPI. |
| Redis | Separate container | Queue metadata, rate limits, playground session state. Not on the token hot path. |
| Prometheus + Grafana | Separate containers | Scrape `/metrics`. |
| Next.js | Separate container / `npm run dev` | Talks to FastAPI only. |

**Do not** run `--workers 4` uvicorn workers each loading the model. That multiplies memory and fights over one GPU. If you later want multi-replica, do it as **one replica per GPU** behind a tiny load balancer, not multiple workers sharing one GPU.

`QUESTION:` Do you have a NVIDIA GPU with CUDA, Apple Silicon (MPS), or CPU only? The architecture stays the same; batch sizes, model choice, and the 200-concurrency claim all change. See Section 18.

---

## 4. Core concepts you must implement (not just read about)

These are the pieces that make the project “inference infrastructure” instead of “FastAPI wrapping `model.generate()`”.

### 4.1 Autoregressive decoding

A decoder-only transformer predicts **one token at a time**. Request state is:

- `input_ids` (prompt + generated so far)
- sampling params (`temperature`, `top_p`, `max_tokens`, `stop`)
- KV cache tensors for every layer
- position / `cache_position` for rotary embeddings

**Naive (forbidden after Phase 2):** call `model(full_sequence)` every step. That recomputes attention over the whole prefix. Cost per new token grows with sequence length.

**Correct:** prefill the prompt once, then decode with `input_ids` of shape `[batch, 1]` and the cached K/V.

### 4.2 Prefill vs. decode (this split is the whole game)

| Phase | Input width | Compute character | GPU behavior |
|---|---|---|---|
| **Prefill** | Full prompt, `seq_len = prompt_len` | Compute-heavy, highly parallel | Often *compute-bound* |
| **Decode** | One token | Memory-heavy: read the whole KV cache + weights | Often *memory-bandwidth-bound* |

A serving system that treats prefill and decode as the same operation will either:

- stall interactive users behind a huge prefill, or
- underutilize the GPU by decoding one request at a time.

Flux should **explicitly separate** them in code (`prefill_step`, `decode_step`) even if they share the same model forward.

### 4.3 KV cache

For each transformer layer, attention computes keys `K` and values `V` from the current tokens. Future tokens attend to all previous K/V. Saving those tensors is the KV cache.

Rough memory (fp16):

```text
kv_bytes ≈ 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * 2_bytes
```

Worked example you should plug into a README later (numbers depend on the actual model — compute them in code, do not hardcode):

- TinyLlama-1.1B: 22 layers, 4 KV heads, head_dim 64  
- 22 * 4 * 64 * 2 (K and V) * 2 bytes ≈ **22 KB per token per request**
- 2048 tokens × 64 concurrent sequences ≈ **~2.8 GB** just for KV, before weights

That arithmetic is why **admission control** exists. You cannot accept unbounded concurrency.

### 4.4 Continuous batching (iteration-level scheduling)

Static batching: wait until `B` requests arrive, run them together, wait until the **longest** one finishes. Short requests pay for long ones. GPU sits idle between batches.

**Continuous batching** (Orca-style):

Every decode iteration:

1. Drop finished sequences.
2. Admit waiting sequences if KV memory / max batch allow it.
3. Prefill newly admitted prompts (or a bounded number of them).
4. Decode **one token** for every active sequence.
5. Stream those tokens out.
6. Repeat.

Requests join and leave the running batch at iteration boundaries. This is the single highest-value feature in the project.

Paper to actually read: *Orca: A Distributed Serving System for Transformer-Based Generative Models* (Yu et al., OSDI 2022). You are implementing the **iteration-level scheduling idea**, not Orca’s distributed runtime.

### 4.5 Paged / blocked KV cache (simplified PagedAttention)

vLLM’s insight: KV cache as **non-contiguous blocks**, like virtual memory.

You should implement a **teaching-scale** version:

- Fixed block size, e.g. 16 tokens.
- A global free-block pool.
- Each request holds a list of block IDs.
- Attention reads K/V by gathering those blocks (PyTorch indexing is fine; no custom CUDA).

This gives you:

- less fragmentation than one giant `[max_batch, max_seq, ...]` tensor
- a real `can_admit(prompt_len, max_new)` check
- a story for prefix caching later (share blocks for common prefixes — stretch)

Do **not** try to match vLLM kernel performance. The data structure is the lesson.

### 4.6 Scheduling and fairness

Implement **at least two policies** and benchmark them:

| Policy | Behavior | Typical effect |
|---|---|---|
| `fcfs` | Admit in arrival order | Simple, can head-of-line block on a huge prompt |
| `memory_fit` | Admit the next request that fits remaining KV blocks | Higher utilization |
| `shortest_prefill_first` (optional) | Prefer short prompts | Better TTFT under mixed load |

Add **admission control**: reject or 429 when the queue exceeds `max_waiting` or estimated TTFT exceeds a SLO.

### 4.7 Streaming

Use **Server-Sent Events (SSE)** for token streaming. It maps cleanly to the OpenAI `stream=true` chunk format and is easier than WebSockets for this use case.

Decode loop produces a token → engine pushes onto a per-request `asyncio.Queue` → FastAPI `StreamingResponse` yields `data: {...}\n\n`.

### 4.8 Observability for ML systems

Track both **HTTP metrics** and **token metrics**. GPU-serving dashboards that only show request/sec are lying to you.

Must-have metrics:

- `flux_requests_in_flight`
- `flux_queue_depth`
- `flux_ttft_seconds` (time to first token) — histogram
- `flux_e2e_latency_seconds` — histogram
- `flux_tokens_generated_total`
- `flux_prefill_tokens_total`
- `flux_decode_tokens_per_second` (gauge)
- `flux_kv_blocks_used` / `flux_kv_blocks_total`
- `flux_batch_size_decode` (gauge)
- `flux_prefill_duration_seconds`
- `flux_decode_step_duration_seconds`

---

## 5. Suggested repository layout

Do not over-engineer packages. One Python package plus one Next app is enough.

```text
flux/
  README.md                          # how to run, architecture sketch, measured numbers
  IMPLEMENTATION_PLAN.md             # this file
  pyproject.toml
  docker-compose.yml
  Dockerfile                         # API + engine
  Makefile                           # dev shortcuts
  .env.example

  src/flux/
    __init__.py
    config.py                        # pydantic-settings: model id, max_batch, block_size, device
    server/
      app.py                         # FastAPI factory
      routes_openai.py               # /v1/chat/completions, /v1/completions
      routes_admin.py                # /health, /admin/stats, /admin/config
      sse.py                         # SSE helpers, OpenAI chunk shaping
    engine/
      types.py                       # Request, Sequence, SamplingParams, SequenceStatus
      tokenizer.py                   # chat template + encode/decode wrapper
      model_loader.py                # HF load, dtype, device, optional torch.compile
      kv_cache.py                    # BlockPool, SequenceBlockTable
      scheduler.py                   # policies + admission
      batching.py                    # build padded / packed batch tensors
      worker.py                      # the loop: admit → prefill → decode → stream
      sampler.py                     # temperature, top_p, greedy, eos/stop
      naive_engine.py                # Phase 1 baseline (kept for A/B benches)
    metrics/
      prometheus.py                  # counters, histograms, /metrics
    redis_client.py                  # optional control plane

  benchmarks/
    loadgen.py                       # concurrent clients, Poisson or closed-loop
    scenarios.yaml                   # shared / mixed / burst traces
    compare_naive_vs_flux.py         # the resume-number script
    report.py                        # markdown + plots (ttft, tpot, throughput)

  tests/
    test_kv_cache.py
    test_scheduler.py
    test_sampler.py
    test_openai_api.py               # httpx + tiny model or mocked logits
    test_continuous_batching.py      # finish times interleave correctly
    test_streaming.py

  dashboard/                         # Next.js + TypeScript
    app/
      page.tsx                       # playground
      metrics/page.tsx               # live charts (or iframe Grafana)
      queue/page.tsx                 # in-flight + waiting visualization
    lib/api.ts

  grafana/
    dashboards/flux.json
    provisioning/...
  prometheus/
    prometheus.yml

  docs/
    architecture.md                  # diagrams you can paste into a report
    papers.md                        # short notes on Orca / PagedAttention
    benchmark_results.md             # filled in after Phase 8
```

---

## 6. Tech stack — what each piece is *for*

| Piece | Role in Flux | Undergrad-honest usage |
|---|---|---|
| **Python 3.11+** | Engine + API | Type hints, asyncio, pytest |
| **PyTorch** | Model forward, tensors on CUDA/CPU/MPS | `torch.inference_mode()`, SDPA, optional `torch.compile` |
| **Hugging Face Transformers** | Weights, tokenizer, chat template | Prefer `use_cache=True` only in the naive baseline; in Flux you own the cache tensors |
| **FastAPI + uvicorn** | HTTP, SSE, OpenAPI | Single worker, async routes |
| **CUDA** | Optional accelerator | Highly recommended. If absent, develop on tiny models and document CPU numbers separately |
| **Redis** | Control plane | Job ids for non-streaming requests, simple rate limit, optional prefix-cache index. **Not** a replacement for the in-memory scheduler |
| **Docker Compose** | One-command demo | api, redis, prometheus, grafana, dashboard |
| **Next.js + TypeScript** | Operator + playground UI | Streaming fetch, queue viz, metric cards |
| **Prometheus + Grafana** | Observability | Scrape FastAPI `/metrics` (prometheus-fastapi-instrumentator or raw `prometheus_client`) |

### Libraries to use, not reimplement

- Tokenizers: Hugging Face `tokenizers` / `transformers.AutoTokenizer`
- Attention kernels: PyTorch scaled_dot_product_attention (uses FlashAttention-2 when available — that **is** allowed)
- HTTP: FastAPI, httpx in tests/loadgen
- Settings: `pydantic-settings`

### Libraries **not** to depend on for the core engine

- vLLM, TGI, TensorRT-LLM, llama.cpp server, Ollama  
  Using them as a **comparison baseline** in a stretch benchmark is fine. They must not be the Flux engine.

---

## 7. API contracts (build these early, keep them stable)

### 7.1 Chat completions (primary)

`POST /v1/chat/completions`

```json
{
  "model": "flux-qwen-0.5b",
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain KV cache in two sentences."}
  ],
  "max_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true,
  "stop": ["<|endoftext|>"]
}
```

Streaming chunks should be **good enough** that the OpenAI JS SDK or a simple `fetch` reader works. Match:

- `choices[0].delta.content`
- final chunk with `finish_reason`
- `data: [DONE]`

Non-streaming response should include `usage: {prompt_tokens, completion_tokens, total_tokens}`.

### 7.2 Completions (optional but easy)

`POST /v1/completions` with `prompt` string or token list. Useful for loadgen so you do not have to build chat templates in the bench client.

### 7.3 Health and ops

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process up, model loaded, device |
| `GET /ready` | Engine loop running, KV pool allocated |
| `GET /metrics` | Prometheus text |
| `GET /admin/stats` | JSON snapshot: queue depth, batch size, KV used, running request ids |
| `POST /admin/abort/{request_id}` | Cancel in-flight (good systems homework) |

### 7.4 Errors

Use HTTP honestly:

- `400` invalid sampling / empty messages
- `404` unknown model name (even if you only serve one)
- `429` queue full / rate limited — include `Retry-After`
- `503` model not loaded
- `504` if you impose a max wait (optional)

---

## 8. Internal data model

Keep these types small and explicit. They are the backbone of tests.

```text
SamplingParams
  temperature: float
  top_p: float
  max_tokens: int
  stop_token_ids: list[int]
  ignore_eos: bool = False

SequenceStatus = WAITING | PREFILL | DECODING | FINISHED | ABORTED | ERROR

Sequence
  id: str
  arrival_ns: int
  prompt_ids: Tensor[int]
  output_ids: list[int]
  status: SequenceStatus
  sampling: SamplingParams
  block_table: list[int]          # KV block ids
  num_computed_tokens: int        # prefill progress if you chunk prefill
  token_queue: asyncio.Queue      # for SSE
  finished_event: asyncio.Event

SchedulerOutput
  prefill_seqs: list[Sequence]
  decode_seqs: list[Sequence]
  preempted: list[Sequence]       # optional: evict to make room

EngineStats
  waiting, running, kv_used, kv_total, last_batch_size, tokens_per_sec
```

**Finish reasons** to implement: `stop` (EOS or stop string), `length` (hit max_tokens), `abort`.

---

## 9. Phased implementation

Estimated effort is in **focused work sessions**, not calendar weeks. A motivated undergrad can complete Phases 0–8 as a substantial semester project; 9–10 are polish and stretch.

---

### Phase 0 — Scaffolding and “hello GPU”

**Goal:** Empty repo becomes a runnable skeleton. You know what device you have.

**Build**

- `pyproject.toml` with: `torch`, `transformers`, `accelerate`, `fastapi`, `uvicorn`, `pydantic-settings`, `prometheus-client`, `redis` (optional extra), `httpx`, `pytest`, `numpy`.
- Pin versions. PyTorch/CUDA mismatch is the #1 time sink.
- `flux config` / `GET /health` printing: device, CUDA capability, VRAM, model id (not loaded yet).
- Docker Compose file with Redis / Prometheus / Grafana stubs (can be unused until later).
- `Makefile`: `make install`, `make test`, `make api`, `make bench`.
- Device selection helper: `cuda` if available else `mps` else `cpu`.

**Do not build** the engine yet.

**Acceptance**

- `pytest` runs (maybe one dummy test).
- A 10-line script prints `torch.cuda.is_available()` and allocates a 1000×1000 tensor on the chosen device.

**Learning checkpoint:** You can explain dtype (`fp16`/`bf16`/`fp32`) and why CPU fp16 is often a trap.

---

### Phase 1 — Naive baseline engine (keep forever)

**Goal:** A correct, slow server you will beat later. This is the control group for every graph.

**Build**

- Load a small chat model with `AutoModelForCausalLM.from_pretrained`.
- `NaiveEngine.generate(prompt, max_tokens)`:
  - tokenize
  - for `_ in range(max_tokens)`: `logits = model(input_ids)` over the **full** sequence, sample, append
  - **do not** pass `past_key_values` yet
- FastAPI `POST /v1/completions` **non-streaming**, one request at a time (a global `asyncio.Lock` is fine).
- Log latency and tokens/sec per request.

**Do not** add a queue yet. Serialization *is* the baseline.

**Acceptance**

- One curl request returns a completion.
- A unit test with a tiny prompt on CPU finishes in reasonable time (`max_tokens=4`, tiny model or mocked).

**Why this phase exists:** Interviewers will ask “compared to what?” You want this code in `naive_engine.py` still importable in Phase 8.

---

### Phase 2 — KV-cached decode (single request)

**Goal:** Stop recomputing the prompt on every token. This is usually the first big speedup on long prompts.

**Build**

- `prefill(prompt_ids) -> logits, kv_cache`
- `decode(last_token, kv_cache) -> logits, kv_cache`
- Use HF `past_key_values` / `DynamicCache` **or** your own tuple of tensors. Either is fine for this phase.
- Measure:
  - TTFT (prefill time)
  - TPOT (time per output token)
  - tokens/sec vs. Phase 1 for prompt lengths 32, 128, 512, 1024

**Acceptance**

- Identical greedy outputs vs. Phase 1 for temperature=0 (bitwise or token-equal).
- Plot: Phase 1 decode time grows with `prompt+t`; Phase 2 decode time is roughly flat.

**Learning checkpoint:** Draw attention with and without cache on a whiteboard. If you cannot, do not proceed.

**Tests**

- Greedy equality vs. naive for 16 tokens.
- KV cache rank/shapes match `n_layers`, `n_kv_heads`, `seq`.

---

### Phase 3 — Sampler, tokenizer pipeline, chat templates

**Goal:** Generation is not `argmax` forever. Chat models need their template or quality looks randomly broken.

**Build**

- `SamplingParams` + `sampler.py`: greedy, temperature, top-p (nucleus). Start with per-sequence sampling even inside a batch (loop is OK at this scale).
- EOS handling and simple stop strings (decode last n chars or match token ids).
- `apply_chat_template` wrapper. Store `prompt_token_ids` after templating so benches are fair.
- Reject `temperature=0` combined with nonsense, or treat `temperature=0` as greedy.

**Acceptance**

- Chat request with a system prompt produces a reply that is clearly following the template (inspect `prompt_ids` in logs).
- Temperature=0 is deterministic across 3 runs.

**Do not** implement beam search, typical decoding, or grammar sampling.

---

### Phase 4 — Request queue, sequence objects, non-batched concurrency

**Goal:** 200 clients can **connect** without 200 model copies. They wait in a queue; the engine still may run one-at-a-time.

This phase is about **control plane**, not GPU efficiency.

**Build**

- `RequestQueue` (asyncio) with `max_waiting`.
- Each HTTP handler creates a `Sequence`, enqueues, waits on `finished_event` or streams from `token_queue`.
- Worker task: `while True: seq = await queue.get(); run_cached_generate(seq)`.
- `429` when waiting queue is full.
- Redis optional: persist `request_id -> status` for the dashboard if the browser polls. In-memory is acceptable if you document it.

**Acceptance**

- `ab` / your loadgen with 50 concurrent requests: all complete or 429; server does not OOM from thread explosion.
- `/admin/stats` shows waiting vs running.

**Learning checkpoint:** Concurrency of **connections** ≠ concurrency of **decode batch**. You now have the first, not the second.

---

### Phase 5 — Continuous batching (the heart of the project)

**Goal:** Multiple sequences share each forward pass. New work joins at iteration boundaries.

**Build**

This is the largest phase. Split it if needed: 5a batch decode of a static set, 5b admit/leave dynamically.

**5a — Static batch decode**

- Pad (or pack) `input_ids` and an attention mask.
- For decode: `input_ids` shape `[B, 1]`, KV cache batched as `[B, ...]`.
- Sequences in a batch must share a **common implementation** of cache layout. Start with **padded contiguous cache**: allocate `max_batch × max_seq` and a boolean `active` mask. Simpler than paging; get correctness first.

**5b — Iteration loop**

Pseudocode you should almost be able to paste into `worker.py`:

```text
loop:
  scheduler.admit_from_waiting(kv_free, max_batch)
  if new_seqs:
      prefill(new_seqs)          # can be one-by-one at first
      stream first token if you generate it during prefill
  if running_seqs:
      tokens = decode_one_step(running_seqs)
      for each seq:
          push token to queue
          if eos or max_tokens: free_cache(seq); mark finished
  else:
      await queue.wait()         # do not busy-spin
  record metrics
```

**Prefill policy (pick a default):**

- **Default:** prefill newly admitted sequences **one at a time** (or a small prefill batch), then join the decode batch. Easier memory accounting.
- **Better:** batched prefill of several prompts. Do this once one-at-a-time works.

**Correctness traps (budget time for these)**

- Position ids / RoPE: wrong cache positions → garbage after the first few tokens.
- Attention mask when padding decode batch.
- Left-pad vs right-pad. HF generate often left-pads; your decode-only batch of length 1 does not care, but **prefill padding** does.
- Finished sequences leaving holes in the batch — compact or use an active mask.
- `torch.inference_mode()` and no accidental autograd graphs.

**Acceptance**

- Two requests of different lengths; the short one **finishes and streams** while the long one is still going (prove with timestamps in logs).
- Greedy outputs match single-request Phase 2 for the same prompt (no batch interference).
- Throughput (output tokens/sec, aggregate) rises vs. Phase 4 under 8–32 concurrent clients.

**Tests**

- Interleaving finish times.
- Token equality vs. solo run.
- Max-batch enforcement.

This phase is where “3x throughput” usually appears, **if** the GPU was idle in the baseline. On CPU the win may be smaller; still report it honestly.

---

### Phase 6 — Blocked KV cache + memory-aware admission

**Goal:** KV memory is a pool with a budget. The scheduler can answer “does this request fit?”

**Build**

- `BlockPool(num_blocks, block_size, num_layers, ...)`
- `allocate(seq, num_tokens) -> list[block_id] | None`
- `free(seq)`
- Copy or scatter K/V into blocks after prefill/decode. A simple approach: keep a contiguous working buffer for the current batch, write back into blocks. A cleaner approach: index `K[:, block_table, ...]` during attention — harder with stock HF models.

**Pragmatic default for HF models:**  
You often **cannot** easily inject a custom paged attention into an arbitrary HF model without writing a custom attention module. For an undergrad project, do this:

1. Implement the **block allocator and accounting** for real (this is the scheduler-facing API).
2. Use a **contiguous cache per sequence** or a padded batch cache **backed by the allocator’s budget** (`num_blocks * block_size >= sum(seq_lens)`).
3. Optionally, for **one** small educational model (see Phase 1 optional toy decoder), implement gather-based paged attention so you can say you did it in model code.

That split is honest and still high value. Pretending you implemented PagedAttention inside Llama while still calling `model(input_ids, past_key_values=...)` is not.

`QUESTION:` Are you okay with “block allocator + HF cache” as the default, plus a tiny custom decoder that does real gather-based paging? That is the recommended path. Full HF attention monkey-patching is a rabbit hole.

**Admission control**

```text
needed_blocks = ceil((prompt_len + max_tokens) / block_size)
# or conservative: prompt_len + max_tokens, you can grow later
if pool.free_blocks < needed_blocks: keep waiting (or preempt)
```

**Preemption (optional in this phase):** if a long decode is hogging memory and many short jobs wait, swap one sequence’s KV to CPU. This is a 80/20 version of vLLM preemption. Only do it if the happy path is solid.

**Acceptance**

- Configure a tiny pool. A 3rd concurrent request waits until one finishes, then runs. Stats show `kv_blocks_used`.
- No CUDA OOM under a soak test that previously OOM’d with unbounded cache.

---

### Phase 7 — Streaming generation (SSE) + OpenAI chunk format

**Goal:** Playground and `curl -N` show tokens live. TTFT becomes a first-class metric.

**Build**

- `stream=true` path using `sse.py`.
- First event as soon as the first decode (or prefill-last-token) is ready — do not buffer the full answer.
- `AbortController` / client disconnect → `SequenceStatus.ABORTED` → free KV. This is a real production bug if skipped (cache leak).

**Acceptance**

- `curl -N` prints tokens incrementally.
- Kill the client; `/admin/stats` running count drops and KV is freed (test this).
- Dashboard playground (Phase 9 can consume this immediately if you stub UI earlier).

---

### Phase 8 — Benchmark harness and the resume numbers

**Goal:** A repeatable experiment, not a vibe.

**Build `benchmarks/loadgen.py`**

Closed-loop and open-loop both teach something. Implement **closed-loop** first (N concurrent workers, each sends the next request when done — easy). Open-loop Poisson arrivals are a stretch but look great.

**Metrics to record per request**

- `ttft_ms` — first token
- `e2e_ms` — last token / `finish_reason`
- `tpot_ms` — (e2e − ttft) / (output_tokens − 1)
- `output_tokens`, `prompt_tokens`
- `http_status`

**Aggregate**

- throughput: `sum(output_tokens) / wall_clock`
- request/sec
- p50 / p95 / p99 for TTFT and e2e
- ignore warmup (first 10–20 requests)

**Scenarios (put in `scenarios.yaml`)**

| Name | Prompt | Output | Concurrency | Why |
|---|---|---|---|---|
| `short_chat` | ~64 tok | 64 | 1, 8, 32, 128, 200 | interactive |
| `long_prompt` | ~1024 tok | 32 | 8, 32 | prefill-heavy |
| `mixed` | 50/50 | 32–256 | 32, 128 | continuous batching shines |
| `naive_vs_flux` | same as short_chat | 64 | 32 | **the resume graph** |

**Methodology rules (write these in `docs/benchmark_results.md`)**

- Same model, same dtype, same machine, same prompts (seed a prompt pool).
- Greedy decoding for A/B correctness; a second run with temperature 0.7 for “realistic.”
- Pin CPU frequency / note background load if you can; at least close other GPU apps.
- Run ≥3 trials; report mean and that you did.

**How to get a 45% p99 / 3x throughput story without lying**

Compare **Phase 1 naive sequential** vs **Phase 5+ continuous batching + KV cache** on the mixed or short_chat scenario at the concurrency where the GPU still has headroom.

- Throughput 3x is common when the baseline leaves the GPU idle between single-request decodes.
- p99 e2e can **increase** at huge concurrency (queueing). That is not a failure — **report both TTFT and e2e**. The resume line should specify **which** latency. Recommendation: “cut p99 **TTFT**” or “cut p99 **TPOT**” rather than a vague “latency.”
- If 200 concurrent on your hardware just queues for seconds, either use a smaller model, shorter `max_tokens`, or change the resume number to whatever concurrency still meets a SLO (e.g. p99 TTFT < 2s).

**Acceptance**

- One command produces a markdown table + PNG/SVG plots committed under `docs/benchmark_results.md`.
- You can explain every column.

---

### Phase 9 — Observability (Prometheus + Grafana)

**Goal:** Watch the scheduler while a loadgen runs. This is how you debug “why is TTFT spiking.”

**Build**

- Instrument the engine with `prometheus_client`.
- `/metrics` endpoint.
- `prometheus.yml` scrape interval 2–5s (token metrics move fast; 15s is sluggish for demos).
- Grafana dashboard JSON in-repo:
  - in-flight / waiting
  - TTFT heatmap or histogram
  - decode tokens/sec
  - KV pool utilization
  - decode batch size
  - prefill vs decode time share

**Acceptance**

- `docker compose up` → Grafana shows moving panels during `make bench`.
- Screenshot goes in the README (this is portfolio gold).

---

### Phase 10 — Next.js dashboard (playground + operator view)

**Goal:** A UI that makes the system *visible*. Not a ChatGPT clone.

**Pages**

1. **Playground**
   - Model name, temperature, max tokens, stream toggle
   - Chat transcript with SSE
   - Side panel: TTFT, token count, request id
2. **Live engine**
   - Poll `/admin/stats` every 500ms or SSE
   - Visual: waiting queue (list), running batch (chips), KV bar
   - This page is the demo you open in interviews
3. **Bench / history (lightweight)**
   - Upload or fetch last `benchmarks/last_run.json`
   - Show the naive vs Flux table

**Stack choices**

- App Router, TypeScript, Tailwind (or shadcn if you want it pretty without custom CSS rabbit holes)
- No auth. Local/demo only.
- Config `NEXT_PUBLIC_FLUX_API=http://localhost:8000`

**Do not build:** user accounts, conversation database, RAG, tool calling UI.

**Acceptance**

- Generate a streamed reply.
- Start loadgen in another terminal; Live engine page shows batch size and KV rising.

---

### Phase 11 — Docker-compose demo path

**Goal:** Someone else can run the interesting parts without matching your conda env.

**Build**

- API image: CPU-friendly default model so Compose works on laptops.
- GPU image/docs: `docker compose --profile gpu` with NVIDIA runtime notes.
- Services: `api`, `dashboard`, `redis`, `prometheus`, `grafana`.
- Volumes for HF cache so you do not re-download 1GB every boot.
- `.env.example` with `FLUX_MODEL`, `FLUX_DEVICE`, `FLUX_MAX_BATCH`, `FLUX_KV_BLOCKS`.

**Acceptance**

- README: 10-line quickstart. Fresh clone → compose → playground works (after model download).

**Note:** GPU-in-Docker is OS-specific. Document “native `make api` for CUDA, Compose for CPU demo” if that is more honest.

---

### Phase 12 — Stretch goals (pick 1–2, not all)

Only after Phases 0–10 work and are measured.

| Stretch | Why it is cool | Risk |
|---|---|---|
| **Prefix / prompt cache** | Share KV blocks for identical system prompts | Correctness of block sharing |
| **Chunked prefill** | Long prompts do not stall decode for 500ms | More scheduler complexity |
| **CUDA graphs** for decode | Lower kernel-launch overhead on NVIDIA | Brittle shapes; skip if no NVIDIA |
| **`torch.compile`** decode step | Easy win sometimes | First-run compile latency |
| **Speculative decoding** with a 1-layer or tiny draft model | Great talking point | Hard to get speedup on small models |
| **OpenAI SDK compatibility test** | `from openai import OpenAI; base_url=...` | Scope creep on API fields |
| **Multi-replica mock load balancer** | Two CPU replicas + nginx, sticky optional | Dilutes the single-engine story |
| **Tiny custom decoder (educational)** | Real paged attention you wrote | Do not let it replace the HF demo |
| **Quantization** (`bitsandbytes` 8-bit / 4-bit) | Memory vs quality tradeoff | Another variable in benches — isolate it |
| **Compare vs vLLM** on same GPU | Humility and a ceiling | Install pain; optional appendix |

Recommended stretch pair: **chunked prefill** + **prefix cache** for a shared system prompt. Both reinforce KV-as-memory.

---

## 10. Naive vs Flux — what you should be able to draw

```text
Naive (Phase 1)
  req1 ████████████████░░░░░░░░░░░░░░░░  (prefill+decode, full recompute)
  req2                 ████████████████
  GPU  [busy][idle wait for HTTP][busy]

Queued, no batch (Phase 4)
  req1 ████████
  req2     wait ████████
  GPU  fully used by 1 seq, others wait  → good utilization per seq, bad throughput

Continuous batch (Phase 5)
  req1  PDDDDDDDDD
  req2    P DDDDDDDDDDDD
  req3         P DDDD
  GPU  [prefill][decode batch of 2-3 every step]
```

P = prefill, D = decode step. The picture of **D columns stacking** is the product insight.

---

## 11. Testing strategy

You will not unit-test CUDA kernels. You will test **invariants**.

| Layer | Tools | What |
|---|---|---|
| Sampler / scheduler / block pool | pytest, CPU | Pure logic, fast |
| KV shapes / greedy equality | pytest, tiny model | 5–10s tests, skip if no weights in CI |
| API | `httpx.AsyncClient` + ASGI | stream chunks parse, 429, abort |
| Engine loop | pytest-asyncio | two sequences, short finishes first |
| Load | `benchmarks/loadgen.py` | not in default CI; `make bench` |

**CI recommendation:** GitHub Actions runs unit tests **without** downloading 1B models. Mock `logits` with a `FakeLM` that returns a fixed next-token id. Keep one optional nightly / local job that runs greedy equality on TinyLlama.

**FakeLM pattern (do this — it unlocks tests):**

```text
class FakeLM:
    n_layers, n_kv_heads, head_dim, vocab
    def prefill(ids): return fake_logits, fake_kv
    def decode(ids, kv): return fake_logits, grown_kv
```

The scheduler and SSE layers should be testable against FakeLM. The HF model is an integration adapter.

---

## 12. Configuration surface

`src/flux/config.py` (env-overridable):

```text
FLUX_MODEL=Qwen/Qwen2.5-0.5B-Instruct
FLUX_DTYPE=auto              # bf16 on Ampere+, fp16, fp32 on CPU
FLUX_DEVICE=auto
FLUX_MAX_BATCH_SIZE=32
FLUX_MAX_WAITING=256
FLUX_BLOCK_SIZE=16
FLUX_NUM_KV_BLOCKS=1024      # derived from VRAM if auto
FLUX_MAX_SEQ_LEN=2048
FLUX_SCHEDULER=fcfs          # fcfs | memory_fit
FLUX_ENABLE_CUDA_GRAPHS=false
FLUX_REDIS_URL=redis://localhost:6379/0
FLUX_ENABLE_REDIS=false      # default off until Phase 4+ wants it
```

Derive `NUM_KV_BLOCKS` from a simple heuristic when `auto`:

```text
budget = 0.3 * usable_vram   # leave room for weights + activations
num_blocks = budget / block_bytes
```

Print the derivation at startup. That log line is a talking point.

---

## 13. Hugging Face usage guidelines (stay in control)

- Load with `AutoModelForCausalLM`, `torch_dtype`, `device_map={"": device}` for a **single** device. Avoid automatic multi-GPU `device_map="auto"` until you understand it (it is not the same as a serving replica).
- Prefer `attn_implementation="sdpa"` (or `"flash_attention_2"` if installed).
- For Flux’s decode loop, call the model with `use_cache=True` and your cache object, `logits` only for the last position.
- Do **not** call `model.generate()` in the optimized path. Using it in naive baseline or as an extra correctness oracle is fine.
- Trust chat templates: `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`.
- Cache weights in `~/.cache/huggingface` and document `HF_HOME` for Docker.

**License:** pick models with a license you can use (Qwen2.5 Apache 2.0, Llama 3.2 has an acceptable use policy — read it). Default to Qwen2.5-0.5B-Instruct to avoid gated downloads blocking Compose.

---

## 14. Redis: use it, but do not fake a distributed GPU

Redis is on the tech stack, so use it for something real:

**Good uses**

- Rate limiting by IP (`INCR` + TTL)
- Non-streaming job store: `POST` returns `request_id`, client polls `GET /v1/requests/{id}` (async completions)
- Dashboard presence if you run API and a worker separately later
- Prefix-cache key index (stretch)

**Bad uses**

- Putting tensors in Redis
- Using Redis as the token stream (too much latency; use asyncio queues)
- Multi-worker GPU sharing via Redis locks pretending to be a scheduler

**Default:** `FLUX_ENABLE_REDIS=false` for the simplest path; compose still runs Redis so you can flip it on in Phase 4.

---

## 15. Dashboard UX notes (keep it an infrastructure UI)

Visual language: dark, dense, “operator console,” not a consumer chatbot.

Must-show numbers on the Live page:

- Decode batch size (big)
- Queue depth
- KV used / total
- Instantaneous tok/s
- p50 TTFT over last 60s (compute in API from a ring buffer if you do not want Grafana on that page)

Playground is secondary. The Live page is the project.

---

## 16. Learning resources (short list, actually read these)

1. Orca (OSDI 2022) — iteration-level scheduling. Skim architecture, steal the figure.
2. vLLM / PagedAttention paper — virtual memory analogy. You implement a toy version of the *idea*.
3. How NVIDIA talks about prefill vs decode (any recent “LLM inference” blog from them or from Hugging Face TGI docs).
4. FastAPI streaming / SSE docs.
5. Prometheus histogram vs summary (so you do not misuse metrics).

Optional: DistServe (prefill/decode disaggregation) — understand why, do not implement two clusters.

Keep 1-page notes in `docs/papers.md`. That folder is for *you* before interviews.

---

## 17. Suggested demo script (for a TA, internship, or portfolio video)

Total: ~4 minutes.

1. Open Grafana + Live engine page (split screen).
2. Send one playground message; show streaming and TTFT.
3. Run `make bench SCENARIO=naive_vs_flux`. Point at p99 and tok/s table.
4. Run 32 concurrent mixed load. Show batch size oscillating, KV bar filling, short jobs completing while a long job continues.
5. Kill a streaming client; KV drops.
6. Flip scheduler `fcfs` vs `memory_fit` if you have both; show queue behavior.

If you cannot demo step 4, Phase 5 is not done.

---

## 18. Open questions for you

Answer these when you can. Until then, **defaults in brackets** apply.

### Hardware and models

1. **What accelerator do you have?** NVIDIA CUDA / Apple MPS / CPU only / cloud GPU credits (Colab, campus cluster)?  
   **[Default: write the code device-agnostic; optimize and publish numbers for whatever this machine is. Keep a “how I would rerun on an A10” paragraph in the README.]**

2. **How much VRAM / RAM?** This caps model size and `NUM_KV_BLOCKS`.  
   **[Default: develop on 0.5B–1.1B.]**

3. **Are you willing to use a gated Hugging Face model (Llama) that needs `huggingface-cli login`?**  
   **[Default: no — Qwen2.5 Apache 2.0 so Docker/CI is painless.]**

### Product shape

4. **Is OpenAI API compatibility a hard requirement, or is a clean custom API OK with an OpenAI-shaped overlay?**  
   **[Default: OpenAI-shaped subset as in Section 7.]**

5. **Do you need multiple models loaded at once (router), or one model per process?**  
   **[Default: one model. Multi-model is a memory nightmare on a student GPU.]**

6. **Auth / API keys?**  
   **[Default: optional `FLUX_API_KEY` header, off by default. No users table.]**

### Redis and Docker

7. **Must Redis be on the token path, or is control-plane-only OK?**  
   **[Default: control-plane only, flaggable.]**

8. **Is Docker Compose a required demo, or is `make api` + hosted Grafana optional?**  
   **[Default: Compose for redis/prometheus/grafana/dashboard; native process for GPU API.]**

### Academic / portfolio constraints

9. **Is this tied to a course with a report deadline, page limit, or required tech (e.g. must show CUDA)?**  
   **[Default: portfolio/semester project with a 5–8 page `docs/report.md` outline you can turn into a writeup.]**

10. **Team size?** Solo vs 2–3. If a team, split: Engine / API+metrics / Dashboard+benches — but **everyone** should be able to explain continuous batching.  
    **[Default: solo.]**

11. **Are you OK with the honest paged-cache split in Phase 6** (allocator + HF cache, optional toy decoder for real paging)?  
    **[Default: yes. Flag if you want to go deeper and monkey-patch Llama attention.]**

12. **Target concurrency for the resume line:** 200 in-flight connections, or 200 **actively decoding**? Those are different.  
    **[Default: 200 in-flight HTTP requests against a small model, with a decode batch much smaller (e.g. 8–32) and the rest waiting — which is how real servers work. State that clearly on the resume: “sustaining 200+ concurrent requests” is connections/queue; also report max decode batch and aggregate tok/s.]**

13. **Latency number: TTFT, e2e, or TPOT?**  
    **[Default: lead with TTFT and throughput; mention e2e at a stated concurrency. Do not claim e2e p99 dropped if what actually dropped is TPOT.]**

### Stretch

14. **Which stretch goals, if any, do you care about?** Prefix cache, chunked prefill, speculative decoding, vLLM comparison, quantization.  
    **[Default: none until Phase 10 ships; then prefix cache + chunked prefill.]**

---

## 19. Risks and how to not die

| Risk | Mitigation |
|---|---|
| CUDA / PyTorch install eats a week | CPU TinyLlama first; GPU as a flag. Pin versions. |
| “I’ll just use `model.generate` and wrap FastAPI” | Phase 1 is that. If Phase 5 never happens, the project is a thin wrapper. Gate yourself. |
| Custom paged attention vs HF internals | Follow Phase 6 default. |
| Chasing 70B | You will debug OOM instead of scheduling. Stay small. |
| Dashboard rabbit hole | Live stats page first, pretty playground last. |
| Unreproducible bench numbers | Script + frozen prompts + warmup + 3 trials. |
| Client disconnect cache leak | Test abort in Phase 7. |
| uvicorn workers × N | One process. Document it. |
| Flashy resume numbers you cannot defend | Measure, then edit the resume line. |

---

## 20. Definition of done (project-level)

The project is **done** (portfolio-ready) when all of the following are true:

1. A naive baseline and a continuous-batching engine both run.
2. KV cache is used on the fast path (no full-sequence recompute).
3. A scheduler with admission control exists; KV pool stats are real.
4. SSE streaming works; disconnect frees memory.
5. FastAPI serves an OpenAI-shaped chat endpoint.
6. Prometheus + Grafana show TTFT, tok/s, batch size, KV during a load test.
7. Next.js playground + live engine page work.
8. `docs/benchmark_results.md` has **your** numbers, plots, hardware, model, and methodology.
9. README can bring up the stack and explains prefill vs decode vs continuous batching in < 30 lines.
10. Tests cover scheduler, block accounting, greedy equality (FakeLM and/or tiny model), and SSE parsing.

When that list is green, update the resume line with measured figures, for example:

> Built Flux, a Python/FastAPI LLM inference server with iteration-level (continuous) batching, KV-cache reuse, and memory-aware request admission. On {GPU/CPU, model}, sustained {N} concurrent clients and improved aggregate throughput {X}x vs. a sequential `model.forward` baseline while reducing p99 TTFT {Y}%.

That sentence is better than the original because it is specific and defensible.

---

## 21. Recommended build order (checklist)

Use this as a kanban. Do not skip the baseline.

- [ ] Phase 0  Scaffold + device probe
- [ ] Phase 1  Naive engine + locked single-request API
- [ ] Phase 2  KV-cached prefill/decode + equality tests
- [ ] Phase 3  Sampler + chat template
- [ ] Phase 4  Queue, 429, admin stats
- [ ] Phase 5  Continuous batching loop
- [ ] Phase 6  Block pool + admission
- [ ] Phase 7  SSE + abort/free
- [ ] Phase 8  Loadgen + naive vs Flux report
- [ ] Phase 9  Prometheus / Grafana
- [ ] Phase 10 Next.js live + playground
- [ ] Phase 11 Compose quickstart
- [ ] Phase 12 One stretch goal
- [ ] Rewrite resume numbers from `docs/benchmark_results.md`

---

## 22. Mapping tech concepts → code you will write

| Concept | Primary code |
|---|---|
| Transformer inference | `model_loader.py`, worker forward |
| Autoregressive decoding | `worker.py` decode step |
| Tokenization pipelines | `tokenizer.py` |
| KV cache management | `kv_cache.py` |
| Dynamic / continuous batching | `worker.py` + `batching.py` |
| Request scheduling | `scheduler.py` |
| Streaming generation | `sse.py`, per-seq queues |
| Throughput vs latency | `benchmarks/*`, Grafana |
| GPU memory management | block pool, dtype, max batch |
| Model serving infrastructure | FastAPI + engine process model |
| Load balancing | out of scope except stretch replica |
| Async processing | asyncio engine loop, FastAPI |
| Resource allocation | admission control |
| Performance benchmarking | `loadgen.py` |
| Observability for ML systems | `metrics/prometheus.py` + Grafana |
| AI infrastructure engineering | the whole repo, especially Phases 5–9 |

---

## 23. One-page architecture decision record (defaults)

If you never answer the questions, build this:

- **Engine:** custom asyncio loop, HF causal LM, one replica.
- **Batching:** continuous, padded decode batch, sequential prefill then join.
- **KV:** block allocator for memory accounting; HF `DynamicCache` or padded tensor as storage.
- **API:** OpenAI-shaped chat + completions, SSE streams.
- **Queue:** in-process asyncio; Redis optional for job status and rate limits.
- **UI:** Next.js playground + live KV/queue view.
- **Metrics:** Prometheus histograms for TTFT/e2e, gauges for batch and KV.
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`.
- **Resume metrics:** measured in Phase 8; original 200 / 45% / 3x is the *target shape*, not a promise.

Start at Phase 0. Do not install vLLM “just to peek” until your own loop generates a token. Curiosity is fine; copying the serving layer is not the assignment.
