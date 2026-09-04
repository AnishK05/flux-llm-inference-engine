# Flux — LLM Inference Infrastructure

## Implementation Plan

This document is the build plan for **Flux**, a production-style LLM inference platform. The goal is not to train a model or invent a new architecture. The goal is to take an already-trained transformer and turn it into a **reliable serving system**: load it, queue requests, batch work, manage KV cache, stream tokens, measure latency vs. throughput, and explain the tradeoffs.

This is an undergrad SWE learning project. The scope is **high-value and interview-ready**, not a vLLM clone and not a multi-cluster production deployment.

**Hardware constraint (locked):** development and all published numbers happen on a **Windows laptop with CPU only**. There is no NVIDIA GPU in the loop. The architecture is still the one used in real GPU serving (prefill/decode split, continuous batching, KV pool, admission control). That is the point: those ideas are about *scheduling and memory*, not about owning a datacenter GPU.

---

## 0. How to use this document

- Build in the **phase order**. Later phases assume earlier ones exist and have a baseline to compare against.
- Treat the **naive baseline** as a first-class artifact. Resume numbers only mean something if you can show *before vs. after* on the same hardware and the same model.
- Every phase has: goal, what you build, what you do **not** build, learning checkpoint, acceptance criteria, and suggested tests.
- Product decisions are **locked** in [Section 18](#18-locked-decisions). There is no remaining “pick later” list.

**Project name:** Flux  
**API style:** OpenAI-compatible *subset* (`/v1/chat/completions` + `/v1/completions`) plus internal admin/metrics endpoints.  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` only (Apache 2.0, ungated).  
**Device:** CPU (`fp32`).  
**Concurrency target:** 200 **in-flight HTTP requests** (queue + running). Decode batch is small (4–8). That is how real servers talk about concurrency, and it is the only 200-request story that is honest on a laptop CPU.

---

## 1. Project thesis (what you are actually learning)

Modern AI products are usually bottlenecked by **inference**, not training. A trained model sitting in a `.safetensors` file does not serve users. Serving it well requires answering:

1. How do I generate the next token without recomputing the entire prompt every time? (**KV cache**)
2. How do I keep the compute units busy when requests arrive at different times and finish at different lengths? (**continuous / iteration-level batching**)
3. How do I admit work when KV memory is the real scarce resource, not HTTP threads? (**scheduler + admission control**)
4. How do I show the user tokens as they are produced? (**streaming**)
5. How do I know whether a change helped? (**benchmarks + Prometheus/Grafana**)

If you can explain those five with diagrams, code you wrote, and plots from your own harness, this project has done its job.

On a CPU laptop the same questions apply, with different bottlenecks:

- Prefill is **compute-bound** (big matmuls over the prompt). Naive recompute of the full sequence every token is brutal. KV cache is usually the first *large* win, especially on longer prompts.
- Decode is **memory-bandwidth-bound** (weights + KV are read for a `[B, 1]` step). Continuous batching still helps because one batched GEMM (`[B, hidden] @ [hidden, vocab]`) is more efficient than `B` sequential GEMMs in MKL/OpenBLAS, and because waiters are no longer fully stalled behind a single sequence.
- You will **not** saturate a GPU. You will still implement the GPU-shaped serving loop, measure it, and explain the tradeoffs. That is the interview story.

### Resume line this project is aiming at

Original sketch:

> Built a production-style LLM inference server in Python and FastAPI with continuous batching, KV-cache reuse, and request queuing, sustaining 200+ concurrent requests while cutting p99 latency 45% and lifting throughput 3x

**Locked wording rules:**

- “200+ concurrent requests” means **in-flight HTTP connections** (waiting + running), not 200 sequences in the decode batch.
- Always name the baseline: sequential full-sequence `forward` with no KV cache (Phase 1).
- Always name the hardware: CPU laptop, `Qwen2.5-0.5B-Instruct`, `fp32`.
- Lead with **p99 TTFT** (KV cache vs. naive recompute) and **aggregate output tokens/sec** (continuous batching vs. single-sequence cached decode). Do not claim “p99 latency” without saying TTFT vs. e2e vs. TPOT.
- Replace 45% / 3x with **measured** figures after Phase 8. A true 2.1x with a plot beats a fake 3x.

Template to fill in after benches:

> Built Flux, a Python/FastAPI LLM inference server with iteration-level (continuous) batching, KV-cache reuse, and memory-aware request admission. On a CPU laptop serving Qwen2.5-0.5B-Instruct, sustained 200+ concurrent in-flight clients (decode batch 4–8) and improved aggregate throughput {X}x vs. a sequential full-recompute baseline while reducing p99 TTFT {Y}%.

---

## 2. What this project is / is not

### It is

- A **from-scratch serving stack** around Hugging Face Qwen weights (you own the batching, cache, scheduler, and HTTP layer).
- A **measurable** comparison: naive sequential serving vs. KV-cached decode vs. continuous batching, all on the same CPU.
- A **small but real** control plane: queue, backpressure, streaming, metrics, a playground UI.
- A system you can talk about in interviews for 20+ minutes without hand-waving, including “what would change on a GPU.”

### It is not

- Training, fine-tuning, RLHF, or prompt-engineering as the main work.
- A from-scratch GPT trained on Tiny Shakespeare as the product.
- A reimplementation of vLLM, TensorRT-LLM, TGI, or Triton.
- Multi-node tensor/pipeline parallelism, Kubernetes, or multi-tenant billing.
- Custom CUDA kernels, CUDA graphs, FlashAttention-2, bitsandbytes GPU quantization, or a vLLM bake-off (no GPU).
- Full OpenAI API compatibility (tools, vision, assistants, batch API, etc.).
- Loading 7B+ models. The laptop cannot do that usefully.

---

## 3. Locked environment: Windows laptop, CPU only

This section is the constraint that every later phase must respect.

### 3.1 Canonical dev environment: WSL2

**Use WSL2 (Ubuntu) as the primary place you write, run, and benchmark code.** Native Windows Python + Docker is possible and more painful: line endings, `make`, compose networking, and path mounts all fight you.

Recommended setup:

1. Windows 10/11 with **WSL2** and Ubuntu.
2. **Docker Desktop for Windows** with the WSL2 backend, Ubuntu distro enabled.
3. Git, Python 3.11+, Node 20+ **inside Ubuntu**, not in Windows PowerShell.
4. Clone the repo in the Linux filesystem (`~/src/...`), not under `/mnt/c/...`. I/O on `/mnt/c` is slow enough to distort CPU benches.

Makefile, Compose, pytest, and the loadgen all assume a Unix shell. PowerShell is not a first-class target.

If WSL2 is unavailable for a day, you can still `pip install` CPU PyTorch on native Windows and run the API, but do not treat those numbers as the official benches.

### 3.2 PyTorch: CPU wheels only

- Install the **CPU** build of PyTorch. Do not install CUDA wheels “just in case” — they are larger and still run on CPU if no driver exists, but the extra complexity is not worth it.
- Default dtype: **`fp32`**. CPU `fp16` is often slower or numerically worse. `bf16` is not a laptop-CPU default.
- Device helper: `cuda` if present else `mps` else `cpu`, but **config default is `cpu`**. You are not developing against a phantom GPU.
- `attn_implementation="sdpa"` (or `"eager"` if SDPA misbehaves on CPU). Never require `flash_attention_2`.

### 3.3 Model and memory budget

`Qwen/Qwen2.5-0.5B-Instruct`:

- Ungated, Apache 2.0, chat template included.
- ~0.5B params → **~2 GB** resident in `fp32`, plus optimizer-free activations and KV.
- Architecture to use in the KV-size formula (verify at load time; do not hardcode forever): 24 layers, 2 KV heads (GQA), head dim 64.

KV size (fp32):

```text
bytes_per_token ≈ 2 * n_layers * n_kv_heads * head_dim * 4
               ≈ 2 * 24 * 2 * 64 * 4
               ≈ 24 KB per token per request
```

So `max_batch=8` × `max_seq=1024` ≈ **192 MB** of KV — fine. Unbounded 200-way decode at 1024 tokens is **not** fine (~4.8 GB KV on top of 2 GB weights). That is why admission control exists.

**Laptop RAM:** 16 GB is comfortable. 8 GB is tight (OS + browser + model + KV). Conservative defaults assume **8–16 GB**:

| Knob | CPU default | Why |
|---|---|---|
| `FLUX_MAX_BATCH_SIZE` | `8` | Batched GEMM helps; 32-way decode thrashes caches and RAM |
| `FLUX_MAX_SEQ_LEN` | `1024` | 2048 doubles KV and prefills for little demo value |
| `FLUX_MAX_WAITING` | `256` | 200 in-flight soak still fits |
| `FLUX_BLOCK_SIZE` | `16` | Teaching-scale paging |
| `FLUX_NUM_KV_BLOCKS` | derived from RAM | See config section |
| `FLUX_DTYPE` | `fp32` | CPU |
| `FLUX_DEVICE` | `cpu` | Locked |

### 3.4 CPU threading (easy, high value)

The engine is **one process**. PyTorch intra-op threads do the matmuls. Oversubscribing (FastAPI workers × OpenMP threads × CPU cores) makes p99 worse.

At process start:

```text
torch.set_num_threads(min(8, os.cpu_count() or 4))
torch.set_num_interop_threads(1)
```

Expose `FLUX_INTRA_THREADS`. Log the chosen values. Do **not** run uvicorn `--workers 4`.

### 3.5 Docker on this laptop

Compose is **CPU-only**. Services: `api`, `dashboard`, `redis`, `prometheus`, `grafana`.

No `nvidia` runtime, no GPU profile, no CUDA base image. The API image uses the CPU PyTorch wheel.

First model download is ~1 GB; mount `HF_HOME` as a volume so you do not fetch it every `compose up`.

### 3.6 What “success” looks like on CPU

You should expect:

- Playground streaming that is usable (small `max_tokens`, 0.5B model).
- A decode batch of a handful of sequences, not dozens.
- 200 concurrent **connections** mostly sitting in the wait queue during a soak, with the live UI showing queue depth vs. batch size — that *is* the demo.
- Throughput and TTFT wins vs. Phase 1 that you can plot. Absolute tok/s will look like a laptop, not like an A100. That is fine.

---

## 4. Target architecture

Keep a **single-process inference engine** as the hot path. FastAPI accepts HTTP, the engine owns the CPU, Redis is for control-plane state (not the token path), and the dashboard is a client of the API + metrics.

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
              PyTorch CPU              Redis                    Hugging Face
              Qwen 0.5B fp32           job status,              Qwen2.5-0.5B
                                       rate limits              Instruct weights
```

### Process model

| Component | Process | Why |
|---|---|---|
| Inference engine + FastAPI | **One Python process**, asyncio, one model replica | Avoids N copies of a 2 GB fp32 model. The CPU matmul is the bottleneck, not FastAPI. |
| Redis | Separate container | Rate limits + async job status. Not on the token hot path. |
| Prometheus + Grafana | Separate containers | Scrape `/metrics`. |
| Next.js | Separate container / `npm run dev` | Talks to FastAPI only. |

**Do not** run `--workers 4` uvicorn workers each loading the model. That multiplies RAM and oversubscribes the CPU.

---

## 5. Core concepts you must implement (not just read about)

These are the pieces that make the project “inference infrastructure” instead of “FastAPI wrapping `model.generate()`”.

### 5.1 Autoregressive decoding

A decoder-only transformer predicts **one token at a time**. Request state is:

- `input_ids` (prompt + generated so far)
- sampling params (`temperature`, `top_p`, `max_tokens`, `stop`)
- KV cache tensors for every layer
- position / `cache_position` for rotary embeddings

**Naive (forbidden after Phase 2):** call `model(full_sequence)` every step. That recomputes attention over the whole prefix. Cost per new token grows with sequence length. On CPU this is the difference between a demo and a lockup.

**Correct:** prefill the prompt once, then decode with `input_ids` of shape `[batch, 1]` and the cached K/V.

### 5.2 Prefill vs. decode (this split is the whole game)

| Phase | Input width | Compute character | CPU laptop behavior |
|---|---|---|---|
| **Prefill** | Full prompt, `seq_len = prompt_len` | Compute-heavy, highly parallel | Often *compute-bound*; long prompts dominate TTFT |
| **Decode** | One token | Memory-heavy: read weights + KV | Often *bandwidth-bound*; batching amortizes weight reads |

A serving system that treats prefill and decode as the same operation will either stall interactive users behind a huge prefill, or underutilize the CPU by decoding one request at a time.

Flux **explicitly separates** them in code (`prefill_step`, `decode_step`) even if they share the same model forward.

### 5.3 KV cache

For each transformer layer, attention computes keys `K` and values `V` from the current tokens. Future tokens attend to all previous K/V. Saving those tensors is the KV cache.

Use the formula in Section 3.3. Compute the real numbers at startup from `config.json` and print them. That log line is a talking point.

That arithmetic is why **admission control** exists. You cannot accept unbounded concurrency on a laptop.

### 5.4 Continuous batching (iteration-level scheduling)

Static batching: wait until `B` requests arrive, run them together, wait until the **longest** one finishes. Short requests pay for long ones. The CPU sits idle between batches.

**Continuous batching** (Orca-style):

Every decode iteration:

1. Drop finished sequences.
2. Admit waiting sequences if KV memory / max batch allow it.
3. Prefill newly admitted prompts (one at a time by default).
4. Decode **one token** for every active sequence.
5. Stream those tokens out.
6. Repeat.

Requests join and leave the running batch at iteration boundaries. This is the single highest-value feature in the project.

On CPU, the win vs. “cached but one-at-a-time” is usually smaller than the win vs. naive recompute, and **both** graphs belong in the report:

1. Phase 1 naive vs. Phase 2 KV cache (TTFT / TPOT, especially long prompts).
2. Phase 4 single-sequence cached vs. Phase 5 continuous batch (aggregate tok/s).

Paper to actually read: *Orca: A Distributed Serving System for Transformer-Based Generative Models* (Yu et al., OSDI 2022). You are implementing the **iteration-level scheduling idea**, not Orca’s distributed runtime.

### 5.5 Blocked KV cache (simplified PagedAttention)

vLLM’s insight: KV cache as **non-contiguous blocks**, like virtual memory.

**Locked approach (easiest honest version):**

1. Implement a real **block allocator** (`BlockPool`) the scheduler uses for `can_admit`.
2. Store K/V with Hugging Face `DynamicCache` / padded tensors **backed by that budget** (`num_blocks * block_size >= sum(seq_lens)`).
3. Do **not** monkey-patch Qwen attention. Do **not** write a second toy decoder. Those are rabbit holes on a semester CPU project.

You still get:

- less “oops I allocated `[max_waiting, max_seq, ...]` and froze the laptop”
- a real `can_admit(prompt_len, max_new)` check
- a story for prefix caching later (stretch): share block ids for a common system prompt, even if physical tensors start as copies

Do **not** try to match vLLM kernel performance. The data structure and admission policy are the lesson.

### 5.6 Scheduling and fairness

Implement **two** policies and benchmark them:

| Policy | Behavior | Typical effect |
|---|---|---|
| `fcfs` | Admit in arrival order | Simple, can head-of-line block on a huge prompt |
| `memory_fit` | Admit the next request that fits remaining KV blocks | Higher utilization |

Skip shortest-prefill-first unless the two above are done and measured.

Add **admission control**: HTTP `429` when the wait queue exceeds `max_waiting`. Do not swap KV to “CPU” (you are already on CPU). Skip preemption.

### 5.7 Streaming

Use **Server-Sent Events (SSE)** for token streaming. It maps cleanly to the OpenAI `stream=true` chunk format and is easier than WebSockets for this use case.

Decode loop produces a token → engine pushes onto a per-request `asyncio.Queue` → FastAPI `StreamingResponse` yields `data: {...}\n\n`.

### 5.8 Observability for ML systems

Track both **HTTP metrics** and **token metrics**. Dashboards that only show request/sec are lying.

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
- `flux_process_rss_bytes` (CPU stand-in for “GPU memory”)
- `flux_intra_op_threads` (info gauge)

---

## 6. Suggested repository layout

Do not over-engineer packages. One Python package plus one Next app is enough.

```text
flux/
  README.md                          # how to run on WSL2, architecture, measured numbers
  IMPLEMENTATION_PLAN.md             # this file
  pyproject.toml
  docker-compose.yml                 # CPU-only
  Dockerfile                         # CPU PyTorch + API
  Makefile                           # WSL2 / Linux shortcuts
  .gitattributes                     # LF for scripts
  .env.example

  src/flux/
    __init__.py
    config.py                        # pydantic-settings
    server/
      app.py
      routes_openai.py
      routes_admin.py
      sse.py
    engine/
      types.py
      tokenizer.py
      model_loader.py                # Qwen 0.5B, cpu, fp32
      kv_cache.py                    # BlockPool + accounting
      scheduler.py
      batching.py
      worker.py
      sampler.py
      naive_engine.py
    metrics/
      prometheus.py
    redis_client.py

  benchmarks/
    loadgen.py
    scenarios.yaml                   # CPU-sized prompts / concurrency
    compare_naive_vs_flux.py
    report.py

  tests/
    test_kv_cache.py
    test_scheduler.py
    test_sampler.py
    test_openai_api.py
    test_continuous_batching.py
    test_streaming.py

  dashboard/
    app/
      page.tsx
      metrics/page.tsx
      queue/page.tsx
    lib/api.ts

  grafana/
    dashboards/flux.json
    provisioning/...
  prometheus/
    prometheus.yml

  docs/
    architecture.md
    papers.md
    benchmark_results.md
    windows-wsl2.md                  # short setup notes
```

---

## 7. Tech stack — what each piece is *for*

| Piece | Role in Flux | How we use it here |
|---|---|---|
| **Python 3.11+** | Engine + API | Type hints, asyncio, pytest; install inside WSL2 |
| **PyTorch (CPU)** | Model forward | `torch.inference_mode()`, SDPA, thread caps |
| **Hugging Face Transformers** | Qwen weights, tokenizer, chat template | You own the decode loop; do not call `generate()` on the fast path |
| **FastAPI + uvicorn** | HTTP, SSE, OpenAPI | Single worker, async routes |
| **CUDA** | Not used | Code may detect it; defaults and Docker never require it |
| **Redis** | Control plane | Rate limit + non-streaming job status. **Not** the scheduler |
| **Docker Compose** | One-command demo | CPU images: api, redis, prometheus, grafana, dashboard |
| **Next.js + TypeScript** | Operator + playground UI | Streaming fetch, queue viz, metric cards |
| **Prometheus + Grafana** | Observability | Scrape FastAPI `/metrics` |

### Libraries to use, not reimplement

- Tokenizers: `transformers.AutoTokenizer`
- Attention: PyTorch SDPA
- HTTP: FastAPI, httpx in tests/loadgen
- Settings: `pydantic-settings`

### Libraries **not** to depend on

- vLLM, TGI, TensorRT-LLM, llama.cpp server, Ollama (engine)
- `flash-attn`, `bitsandbytes`, CUDA-only extras
- Intel OpenVINO / IPEX (tempting on CPU, extra install maze — skip)

---

## 8. API contracts (build these early, keep them stable)

### 8.1 Chat completions (primary)

`POST /v1/chat/completions`

```json
{
  "model": "flux-qwen-0.5b",
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Explain KV cache in two sentences."}
  ],
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true
}
```

Default `max_tokens` in the playground is **64**, not 128–2048. CPU decode is slow; the UI should feel alive.

Streaming chunks should be good enough that the OpenAI JS SDK or a simple `fetch` reader works:

- `choices[0].delta.content`
- final chunk with `finish_reason`
- `data: [DONE]`

Non-streaming response includes `usage: {prompt_tokens, completion_tokens, total_tokens}`.

Unknown `model` names: accept `flux-qwen-0.5b` and the HF id; `404` anything else.

### 8.2 Completions (loadgen-friendly)

`POST /v1/completions` with a `prompt` string. The bench client should use this so it does not reimplement chat templates.

### 8.3 Health and ops

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process up, model loaded, device=`cpu`, thread counts |
| `GET /ready` | Engine loop running, KV pool allocated |
| `GET /metrics` | Prometheus text |
| `GET /admin/stats` | JSON: queue depth, batch size, KV used, RSS, running ids |
| `POST /admin/abort/{request_id}` | Cancel in-flight |

### 8.4 Errors

- `400` invalid sampling / empty messages
- `404` unknown model name
- `429` queue full / rate limited — include `Retry-After`
- `503` model not loaded
- No auth. Local demo only.

---

## 9. Internal data model

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
  block_table: list[int]
  num_computed_tokens: int
  token_queue: asyncio.Queue
  finished_event: asyncio.Event

SchedulerOutput
  prefill_seqs: list[Sequence]
  decode_seqs: list[Sequence]

EngineStats
  waiting, running, kv_used, kv_total, last_batch_size, tokens_per_sec, rss_bytes
```

Finish reasons: `stop`, `length`, `abort`. No preemption status.

---

## 10. Phased implementation

Estimated effort is in **focused work sessions**, not calendar weeks. Phases 0–8 are the semester core. 9–11 are portfolio polish. Phase 12 is one stretch after that.

---

### Phase 0 — Scaffolding and “hello CPU”

**Goal:** Empty repo becomes a runnable skeleton on WSL2. You know RAM, core count, and that PyTorch is the CPU wheel.

**Build**

- `pyproject.toml`: `torch` (CPU), `transformers`, `accelerate`, `fastapi`, `uvicorn`, `pydantic-settings`, `prometheus-client`, `redis`, `httpx`, `pytest`, `numpy`. Pin versions.
- `.gitattributes` forcing LF on `*.sh` and `Makefile`.
- `docs/windows-wsl2.md`: Docker Desktop WSL2 backend, clone on the Linux filesystem, Python 3.11.
- Health/config print: `device=cpu`, `torch.__version__`, `cpu_count`, `intra_threads`, RSS, model id (not loaded yet).
- Docker Compose stubs for redis / prometheus / grafana (can be unused until later).
- `Makefile`: `make install`, `make test`, `make api`, `make bench`.
- Device helper defaults to CPU.

**Do not build** the engine yet.

**Acceptance**

- `pytest` runs (one dummy test).
- A 10-line script allocates a 1000×1000 fp32 tensor, prints `tensor.device == cpu`, and `torch.cuda.is_available() == False`.

**Learning checkpoint:** dtype (`fp32` vs `fp16` on CPU) and why the CUDA extra index is the wrong install.

---

### Phase 1 — Naive baseline engine (keep forever)

**Goal:** A correct, slow server you will beat later. This is the control group for every graph.

**Build**

- Load `Qwen/Qwen2.5-0.5B-Instruct` with `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float32)`.
- `NaiveEngine.generate(prompt, max_tokens)`:
  - tokenize
  - for `_ in range(max_tokens)`: `logits = model(input_ids)` over the **full** sequence, sample, append
  - **do not** pass `past_key_values`
- FastAPI `POST /v1/completions` **non-streaming**, one request at a time (a global `asyncio.Lock` is fine).
- Cap playground/demo `max_tokens` at 32–64 so a curl does not take a minute.
- Log latency and tokens/sec per request.

**Do not** add a queue yet. Serialization *is* the baseline.

**Acceptance**

- One curl returns a completion on WSL2.
- A unit test with `max_tokens=4` finishes in reasonable time (or use FakeLM in CI; Qwen in local integration).

**Why this phase exists:** Interviewers will ask “compared to what?” Keep `naive_engine.py` importable in Phase 8.

---

### Phase 2 — KV-cached decode (single request)

**Goal:** Stop recomputing the prompt on every token. On CPU this is usually the largest single speedup.

**Build**

- `prefill(prompt_ids) -> logits, kv_cache`
- `decode(last_token, kv_cache) -> logits, kv_cache`
- HF `past_key_values` / `DynamicCache` is the right storage for this phase.
- Measure TTFT, TPOT, tokens/sec vs. Phase 1 for prompt lengths **32, 128, 512** (skip 1024 until it is clearly fast enough; do not freeze the laptop).

**Acceptance**

- Identical greedy outputs vs. Phase 1 for `temperature=0`.
- Plot: Phase 1 per-token time grows with `prompt+t`; Phase 2 decode time is roughly flat.

**Learning checkpoint:** Draw attention with and without cache. If you cannot, do not proceed.

**Tests**

- Greedy equality vs. naive for 8–16 tokens (FakeLM in CI; Qwen locally).
- KV cache rank/shapes match Qwen (`n_layers=24`, `n_kv_heads=2`, … — assert against config, not magic numbers only).

---

### Phase 3 — Sampler, tokenizer pipeline, chat templates

**Goal:** Generation is not `argmax` forever. Qwen-Instruct needs its chat template or quality looks randomly broken.

**Build**

- `SamplingParams` + `sampler.py`: greedy, temperature, top-p. Per-sequence sampling inside a batch (Python loop is fine at `B<=8`).
- EOS handling. Use Qwen’s actual EOS / im_end ids from the tokenizer.
- `tokenizer.apply_chat_template(..., add_generation_prompt=True)`. Store `prompt_token_ids` after templating so benches are fair.
- Treat `temperature=0` as greedy.

**Acceptance**

- Chat request with a system prompt; logs show templated special tokens, not raw user text alone.
- Temperature=0 is deterministic across 3 runs.

**Do not** implement beam search, typical decoding, or grammar sampling.

---

### Phase 4 — Request queue, sequence objects, non-batched concurrency

**Goal:** 200 clients can **connect** without 200 model copies. They wait in a queue; the engine may still run one-at-a-time.

This phase is the **control plane**, not GEMM efficiency.

**Build**

- `RequestQueue` (asyncio) with `max_waiting=256`.
- Each HTTP handler creates a `Sequence`, enqueues, waits on `finished_event` or streams from `token_queue`.
- Worker: `while True: seq = await queue.get(); run_cached_generate(seq)`.
- `429` when the waiting queue is full.
- In-memory status is enough. Redis stays off until Compose (Phase 11) flips `FLUX_ENABLE_REDIS` for rate limits / job ids.

**Acceptance**

- Loadgen with 50 concurrent requests: all complete or 429; RSS does not explode from thread storms.
- `/admin/stats` shows waiting vs running.

**Learning checkpoint:** Concurrency of **connections** ≠ concurrency of **decode batch**.

---

### Phase 5 — Continuous batching (the heart of the project)

**Goal:** Multiple sequences share each forward pass. New work joins at iteration boundaries.

Split if needed: 5a static batch decode, 5b admit/leave dynamically.

**5a — Static batch decode**

- Pad `input_ids` and an attention mask.
- Decode: `input_ids` shape `[B, 1]`, KV batched as `[B, ...]`.
- Start with **padded contiguous cache**: `max_batch × max_seq` and an `active` mask. `max_batch=8`.

**5b — Iteration loop**

```text
loop:
  scheduler.admit_from_waiting(kv_free, max_batch)
  if new_seqs:
      prefill(new_seqs)          # one-at-a-time
      stream first token if produced during prefill
  if running_seqs:
      tokens = decode_one_step(running_seqs)
      for each seq:
          push token to queue
          if eos or max_tokens: free_cache(seq); mark finished
  else:
      await queue.wait()
  record metrics
```

**Prefill policy (locked):** newly admitted sequences are prefilled **one at a time**, then join the decode batch. Batched prefill is a stretch, not the default.

**Correctness traps**

- Position ids / RoPE: wrong cache positions → garbage after a few tokens.
- Attention mask when padding the decode batch.
- Prefill padding (left vs right). Decode of length 1 does not care; prefill does.
- Finished sequences leaving holes — compact or use an active mask.
- `torch.inference_mode()`; no autograd graphs.

**Acceptance**

- Two requests of different lengths; the short one **finishes and streams** while the long one continues (timestamps in logs).
- Greedy outputs match single-request Phase 2 (no batch interference).
- Aggregate output tokens/sec rises vs. Phase 4 under **4–8** concurrent clients (not 32 — that is a GPU number).

**Tests**

- Interleaving finish times.
- Token equality vs. solo run.
- Max-batch enforcement (`B<=8`).

---

### Phase 6 — Block pool + memory-aware admission

**Goal:** KV memory is a pool with a budget. The scheduler can answer “does this request fit?” without paging inside Qwen attention.

**Build**

- `BlockPool(num_blocks, block_size, ...)`
- `allocate(seq, num_tokens) -> list[block_id] | None`
- `free(seq)`
- Physical cache remains HF / padded tensors. The pool is **accounting + admission**.

```text
needed_blocks = ceil((prompt_len + max_tokens) / block_size)
if pool.free_blocks < needed_blocks: keep waiting
```

Skip preemption (KV swap). Skip custom paged attention kernels.

**Acceptance**

- Tiny pool: a 3rd concurrent request waits until one finishes, then runs. Stats show `kv_blocks_used`.
- Soak that previously ballooned RSS now stays within the configured block budget (watch RSS; there is no CUDA OOM).

---

### Phase 7 — Streaming generation (SSE) + OpenAI chunk format

**Goal:** Playground and `curl -N` show tokens live. TTFT becomes a first-class metric.

**Build**

- `stream=true` via `sse.py`.
- First event as soon as the first token is ready — do not buffer the full answer.
- Client disconnect → `ABORTED` → free KV. Skipping this leaks the pool.

**Acceptance**

- `curl -N` prints tokens incrementally (from WSL2; in PowerShell use `curl.exe -N` if you ever test from Windows).
- Kill the client; running count drops and KV is freed.

---

### Phase 8 — Benchmark harness and the resume numbers

**Goal:** A repeatable experiment on **this laptop**, not a vibe and not a GPU blog number.

**Build `benchmarks/loadgen.py`**

Closed-loop first (N concurrent workers, each sends the next request when done). Open-loop Poisson is out of scope.

**Per-request metrics:** `ttft_ms`, `e2e_ms`, `tpot_ms`, `output_tokens`, `prompt_tokens`, `http_status`.

**Aggregates:** output tokens / wall clock, request/sec, p50/p95/p99 TTFT and e2e. Drop warmup (first 10 requests; benches are short on CPU).

**Scenarios (CPU-sized)**

| Name | Prompt | Output | Concurrency | Why |
|---|---|---|---|---|
| `short_chat` | ~64 tok | 32 | 1, 4, 8 | interactive |
| `long_prompt` | ~512 tok | 16 | 1, 4 | prefill / KV-cache win |
| `mixed` | 50/50 short+long | 16–64 | 8 | continuous batching |
| `naive_vs_flux` | short_chat | 32 | **4** | **resume graph** (do not use 32) |
| `soak_200` | ~32 tok | 8 | 200 in-flight | queue + 429 + live UI; not a latency SLO |

Close the browser and other heavy apps during `naive_vs_flux`. Note Windows power plan (plugged in vs. battery) in `docs/benchmark_results.md` — it really does move CPU numbers.

Run ≥3 trials; report mean.

**How to tell the performance story without lying**

1. **TTFT / TPOT:** Phase 1 vs. Phase 2 on `long_prompt`. This is where “cut p99 TTFT ~X%” usually comes from on CPU.
2. **Throughput:** Phase 4 vs. Phase 5 on `naive_vs_flux` or `mixed` at concurrency 4–8. This is “lifting throughput Yx.”
3. **200 concurrent:** `soak_200` proves the control plane (no crash, bounded KV, 429 after `max_waiting`). Do not quote soak e2e p99 as the latency win.

If Yx is 1.4 not 3, print 1.4. The architecture is still the resume; the plot is the evidence.

**Acceptance**

- One command writes `docs/benchmark_results.md` plus PNG/SVG plots.
- Hardware line includes: Windows + WSL2, CPU model, core count, RAM, `FLUX_INTRA_THREADS`, Qwen2.5-0.5B, fp32.

---

### Phase 9 — Observability (Prometheus + Grafana)

**Goal:** Watch the scheduler while a loadgen runs.

**Build**

- `prometheus_client` + `/metrics`.
- Scrape interval 5s (2s is noisy and the laptop is already busy).
- Grafana dashboard in-repo: in-flight / waiting, TTFT histogram, decode tok/s, KV pool, decode batch size, RSS, prefill vs decode time.

**Acceptance**

- `docker compose up` → Grafana panels move during `make bench`.
- Screenshot in the README.

---

### Phase 10 — Next.js dashboard (playground + operator view)

**Goal:** A UI that makes the system *visible*. Not a ChatGPT clone.

**Pages**

1. **Playground** — model name, temperature, max tokens (default 64), stream on. Side panel: TTFT, token count, request id.
2. **Live engine** — poll `/admin/stats` every 500ms. Waiting list, running batch chips, KV bar, RSS. **This is the interview page.**
3. **Bench** — render `benchmarks/last_run.json` naive vs Flux table.

Stack: App Router, TypeScript, Tailwind. No auth. `NEXT_PUBLIC_FLUX_API=http://localhost:8000`.

**Do not build:** accounts, conversation DB, RAG, tool calling.

**Acceptance**

- Stream a reply.
- Start loadgen; Live page shows batch size and KV rising, queue filling on `soak_200`.

---

### Phase 11 — Docker Compose demo path (CPU)

**Goal:** One command on this Windows + Docker Desktop setup.

**Build**

- CPU Dockerfile (no CUDA base).
- Services: `api`, `dashboard`, `redis`, `prometheus`, `grafana`.
- Volume for HF cache.
- `.env.example`: `FLUX_MODEL=Qwen/Qwen2.5-0.5B-Instruct`, `FLUX_DEVICE=cpu`, `FLUX_DTYPE=fp32`, `FLUX_MAX_BATCH=8`, `FLUX_MAX_SEQ_LEN=1024`.
- `FLUX_ENABLE_REDIS=true` **in Compose only** (rate limit + job status). Local `make api` keeps Redis off so pytest stays simple.

**Acceptance**

- README quickstart: WSL2 + Docker Desktop → compose → playground, including “first run downloads ~1 GB.”

No GPU profile. No NVIDIA docs beyond a one-liner: “the engine is device-agnostic; this repo is validated on CPU.”

---

### Phase 12 — Stretch (pick this one)

Only after Phases 0–11 work and are measured.

**Locked stretch: prefix / system-prompt KV reuse.** Identical leading token ids share prefix accounting (and copy-free tensors if that stays simple). This is the highest-value, lowest-drama extra on a CPU laptop: every playground chat shares the same system prompt.

Explicitly **out of scope** on this hardware (do not start):

- CUDA graphs, FlashAttention, bitsandbytes
- Speculative decoding
- vLLM comparison
- Chunked prefill (nice on GPU; extra scheduler complexity)
- Custom paged attention inside Qwen
- Multi-replica load balancer
- `torch.compile` as a required path (optional evening experiment only; compile latency on CPU is annoying)

---

## 11. Naive vs Flux — what you should be able to draw

```text
Naive (Phase 1)
  req1 ████████████████░░░░░░░░░░░░░░░░  (prefill+decode, full recompute)
  req2                 ████████████████
  CPU  [busy][idle wait for HTTP][busy]

Queued, no batch (Phase 4)
  req1 ████████
  req2     wait ████████
  CPU  fully used by 1 seq, others wait  → good per-seq cache, bad throughput

Continuous batch (Phase 5)
  req1  PDDDDDDDDD
  req2    P DDDDDDDDDDDD
  req3         P DDDD
  CPU  [prefill][decode batch of 2-8 every step]
```

P = prefill, D = decode step. The picture of **D columns stacking** is the product insight. On this laptop the stacked D width is 4–8, not 32–128.

---

## 12. Testing strategy

You will test **invariants**, not CUDA kernels.

| Layer | Tools | What |
|---|---|---|
| Sampler / scheduler / block pool | pytest, CPU | Pure logic, fast |
| KV shapes / greedy equality | pytest | FakeLM in CI; optional local Qwen |
| API | `httpx.AsyncClient` + ASGI | stream chunks, 429, abort |
| Engine loop | pytest-asyncio | two sequences, short finishes first |
| Load | `benchmarks/loadgen.py` | not in default CI; `make bench` |

**CI:** GitHub Actions runs unit tests **without** downloading Qwen. Mock logits with `FakeLM`. A local `make test-integration` downloads 0.5B once and runs greedy equality.

```text
class FakeLM:
    n_layers, n_kv_heads, head_dim, vocab
    def prefill(ids): return fake_logits, fake_kv
    def decode(ids, kv): return fake_logits, grown_kv
```

Scheduler and SSE should be testable against FakeLM. Qwen is an integration adapter.

---

## 13. Configuration surface

`src/flux/config.py` (env-overridable):

```text
FLUX_MODEL=Qwen/Qwen2.5-0.5B-Instruct
FLUX_DTYPE=fp32
FLUX_DEVICE=cpu
FLUX_MAX_BATCH_SIZE=8
FLUX_MAX_WAITING=256
FLUX_BLOCK_SIZE=16
FLUX_NUM_KV_BLOCKS=auto
FLUX_MAX_SEQ_LEN=1024
FLUX_SCHEDULER=fcfs          # fcfs | memory_fit
FLUX_INTRA_THREADS=auto      # min(8, cpu_count)
FLUX_REDIS_URL=redis://localhost:6379/0
FLUX_ENABLE_REDIS=false
```

When `FLUX_NUM_KV_BLOCKS=auto`:

```text
# leave room for OS + 2GB weights + activations
budget = 0.20 * usable_ram
num_blocks = budget / block_bytes
cap so that max_batch * max_seq / block_size is the upper bound you actually need
```

Print the derivation at startup.

---

## 14. Hugging Face usage guidelines (stay in control)

- Load `AutoModelForCausalLM` with `torch_dtype=torch.float32` and `.to("cpu")`. No `device_map="auto"`.
- `attn_implementation="sdpa"`.
- Fast path: `use_cache=True`, logits for the last position only. **Do not** call `model.generate()` there. Naive baseline / correctness oracle may.
- Chat template: `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`.
- Cache weights in `~/.cache/huggingface`; document `HF_HOME` for Docker.
- **License:** Qwen2.5 Apache 2.0. No gated Llama, no `huggingface-cli login` required for the happy path.

---

## 15. Redis: control plane only

**Use**

- Rate limiting by IP (`INCR` + TTL) when Compose is up
- Non-streaming job store: `POST` returns `request_id`, poll `GET /v1/requests/{id}`

**Do not**

- Put tensors in Redis
- Stream tokens through Redis
- Pretend Redis is a distributed GPU lock

**Default:** `FLUX_ENABLE_REDIS=false` for `make api` and pytest. Compose sets it true.

---

## 16. Dashboard UX notes

Visual language: dark, dense, “operator console,” not a consumer chatbot.

Live page must show:

- Decode batch size (big)
- Queue depth (this is how you show 200 concurrent)
- KV used / total
- Instantaneous tok/s
- RSS
- p50 TTFT over last 60s (ring buffer in API is enough)

Playground is secondary. The Live page is the project.

---

## 17. Learning resources (short list, actually read these)

1. Orca (OSDI 2022) — iteration-level scheduling. Skim architecture, steal the figure.
2. vLLM / PagedAttention paper — virtual memory analogy. You implement admission + a block pool, not the CUDA kernel.
3. Prefill vs decode (NVIDIA or Hugging Face TGI explainers still apply on CPU).
4. FastAPI streaming / SSE docs.
5. Prometheus histogram vs summary.

Optional: DistServe — understand why, do not implement two clusters.

Keep 1-page notes in `docs/papers.md`.

---

## 18. Locked decisions

These replace the old open-question list. Build against them.

| Topic | Decision |
|---|---|
| Accelerator | **CPU only.** Windows laptop. No CUDA, no MPS, no Colab required. |
| Dev OS | **WSL2 Ubuntu** + Docker Desktop (WSL2 backend). Repo lives on the Linux filesystem. |
| Model | **`Qwen/Qwen2.5-0.5B-Instruct` only.** Apache 2.0, ungated. No Llama. |
| Dtype / device | `fp32` / `cpu` |
| Max decode batch | **8** |
| Max seq len | **1024** |
| API | OpenAI-shaped subset (Section 8). No tools/vision. |
| Models in process | **One.** |
| Auth | **None.** |
| Redis | Control plane only; off by default, on in Compose. |
| Docker | CPU Compose stack; no GPU profile. |
| KV implementation | Block allocator + HF/padded cache. No attention monkey-patch, no toy decoder. |
| Preemption | **No.** |
| Prefill | One-at-a-time, then join decode batch. |
| Concurrency story | **200 in-flight HTTP requests**; decode batch 4–8. |
| Latency headline | **p99 TTFT** vs naive, plus aggregate tok/s vs queued-unbatched. Not vague “p99 latency.” |
| Team | Solo. |
| Stretch | Prefix / system-prompt cache only, after Phase 11. |
| Course extras | Portfolio writeup in `docs/report.md` after benches exist. No CUDA requirement. |

---

## 19. Risks and how to not die

| Risk | Mitigation |
|---|---|
| Fighting native Windows instead of WSL2 | Canonical path is WSL2; document it first in Phase 0 |
| `/mnt/c` clone makes benches look random | Clone under `$HOME` in Ubuntu |
| CPU fp16 / CUDA wheel install maze | Pin CPU torch, `fp32` only |
| 8 GB RAM machine swap-thrashing | Conservative max_batch/max_seq; RSS metric; shrink pool |
| “I’ll just use `model.generate` and wrap FastAPI” | Phase 1 is that. Phase 5 is the project. |
| Custom paged attention vs HF internals | Phase 6 is accounting only |
| Chasing 7B on CPU | Stay on 0.5B |
| Dashboard rabbit hole | Live stats page first |
| Unreproducible benches | Script + frozen prompts + warmup + 3 trials + plugged-in power |
| Client disconnect cache leak | Test abort in Phase 7 |
| uvicorn workers × N | One process |
| Quoting GPU-blog 3x / 200-active-decode | Measure; 200 = in-flight; Yx from your plots |
| Soak test e2e p99 looking terrible | Expected; do not put soak e2e on the resume |

---

## 20. Definition of done (project-level)

The project is **done** (portfolio-ready) when all of the following are true:

1. A naive baseline and a continuous-batching engine both run on CPU Qwen 0.5B.
2. KV cache is used on the fast path (no full-sequence recompute).
3. A scheduler with admission control exists; KV pool stats are real.
4. SSE streaming works; disconnect frees memory.
5. FastAPI serves an OpenAI-shaped chat endpoint.
6. Prometheus + Grafana show TTFT, tok/s, batch size, KV, RSS during a load test.
7. Next.js playground + live engine page work; live page can show a 200-connection soak filling the queue.
8. `docs/benchmark_results.md` has **your** numbers, plots, **CPU model / RAM / WSL2**, Qwen 0.5B, fp32, and methodology.
9. README brings up the stack on Windows+WSL2+Docker Desktop and explains prefill vs decode vs continuous batching in < 30 lines.
10. Tests cover scheduler, block accounting, greedy equality (FakeLM), and SSE parsing.

Then fill the resume template in Section 1 from that doc.

---

## 21. Recommended build order (checklist)

- [ ] Phase 0  WSL2 scaffold + CPU probe
- [ ] Phase 1  Naive engine + locked single-request API
- [ ] Phase 2  KV-cached prefill/decode + equality tests
- [ ] Phase 3  Sampler + Qwen chat template
- [ ] Phase 4  Queue, 429, admin stats
- [ ] Phase 5  Continuous batching loop (`B<=8`)
- [ ] Phase 6  Block pool + admission
- [ ] Phase 7  SSE + abort/free
- [ ] Phase 8  Loadgen + naive vs Flux report (CPU scenarios)
- [ ] Phase 9  Prometheus / Grafana
- [ ] Phase 10 Next.js live + playground
- [ ] Phase 11 CPU Compose quickstart
- [ ] Phase 12 Prefix-cache stretch
- [ ] Rewrite resume numbers from `docs/benchmark_results.md`

---

## 22. Mapping tech concepts → code you will write

| Concept | Primary code |
|---|---|
| Transformer inference | `model_loader.py`, worker forward |
| Autoregressive decoding | `worker.py` decode step |
| Tokenization pipelines | `tokenizer.py` (Qwen chat template) |
| KV cache management | `kv_cache.py` |
| Dynamic / continuous batching | `worker.py` + `batching.py` |
| Request scheduling | `scheduler.py` |
| Streaming generation | `sse.py`, per-seq queues |
| Throughput vs latency | `benchmarks/*`, Grafana |
| Memory management (CPU RSS + KV) | block pool, dtype, max batch |
| Model serving infrastructure | FastAPI + one process |
| Load balancing | out of scope |
| Async processing | asyncio engine loop, FastAPI |
| Resource allocation | admission control |
| Performance benchmarking | `loadgen.py` |
| Observability for ML systems | `metrics/prometheus.py` + Grafana |
| AI infrastructure engineering | the whole repo, especially Phases 5–9 |

---

## 23. Architecture decision record (locked)

- **Hardware:** Windows laptop, CPU only; develop in WSL2.
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`, fp32, one replica.
- **Engine:** custom asyncio loop, HF causal LM.
- **Batching:** continuous; padded decode batch `<=8`; sequential prefill then join.
- **KV:** block allocator for memory accounting; HF `DynamicCache` / padded tensor as storage.
- **API:** OpenAI-shaped chat + completions, SSE streams, no auth.
- **Queue:** in-process asyncio; Redis optional for job status and rate limits.
- **UI:** Next.js playground + live KV/queue view (queue depth is how 200 concurrent is shown).
- **Metrics:** Prometheus histograms for TTFT/e2e, gauges for batch, KV, RSS.
- **Resume metrics:** measured in Phase 8 on this CPU; 200 = in-flight; headline = p99 TTFT + aggregate tok/s.

Start at Phase 0. Do not install vLLM “just to peek” until your own loop generates a token on this laptop. Curiosity is fine; copying the serving layer is not the assignment.
