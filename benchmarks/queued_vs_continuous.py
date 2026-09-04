#!/usr/bin/env python3
"""Compare Phase 4 sequential queued serving vs Phase 5 continuous batching.

Default: FakeLM (CI-safe; timings are not meaningful).
Pass --qwen to load Qwen2.5-0.5B-Instruct and write a real tok/s table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.scheduler import RequestQueue, Scheduler
from flux.engine.sequence import Sequence
from flux.engine.tokenizer import encode_text
from flux.engine.types import SamplingParams
from flux.engine.worker import ContinuousWorker, QueuedWorker
from flux.runtime import apply_thread_caps


def _prompt_of_len(target_tokens: int) -> str:
    return ("alpha " * target_tokens).strip()


async def _throughput(worker, engine, prompts: list[str], max_tokens: int) -> dict:
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    seqs = [
        Sequence(prompt_ids=encode_text(engine.tokenizer, prompt, device=engine.device), sampling=params)
        for prompt in prompts
    ]
    task = asyncio.create_task(worker.run())
    started = time.perf_counter()
    try:
        for seq in seqs:
            worker.scheduler.submit(seq)
        await asyncio.gather(*[seq.finished_event.wait() for seq in seqs])
        wall = time.perf_counter() - started
    finally:
        worker.request_stop()
        await asyncio.wait_for(task, timeout=30)
    tokens = sum(len(seq.output_ids) for seq in seqs)
    errors = sum(1 for seq in seqs if seq.result is None)
    return {
        "engine": worker.engine_name,
        "n": len(seqs),
        "completion_tokens": tokens,
        "wall_s": wall,
        "tok_s": (tokens / wall) if wall > 0 else 0.0,
        "errors": errors,
        "max_batch_seen": worker.stats.last_batch_size,
    }


def _make_worker(kind: str, engine: CachedEngine, max_batch: int):
    scheduler = Scheduler(RequestQueue(256), max_batch=max_batch)
    if kind == "queued":
        return QueuedWorker(engine, scheduler)
    return ContinuousWorker(engine, scheduler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--concurrency", default="4,8")
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--out", default="docs/phase5_queued_vs_continuous.json")
    args = parser.parse_args()
    apply_thread_caps("auto")

    if args.qwen:
        from flux.config import Settings
        from flux.engine.model_loader import load_causal_lm

        settings = Settings(device="cpu", dtype="fp32")
        model, tokenizer = load_causal_lm(settings)
        engine = CachedEngine(model, tokenizer, device="cpu")
        backend = "qwen"
    else:
        engine = CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")
        backend = "fake"

    concs = [int(x) for x in args.concurrency.split(",") if x.strip()]
    prompt = _prompt_of_len(args.prompt_len)
    rows = []
    for n in concs:
        prompts = [prompt] * n
        queued = asyncio.run(_throughput(_make_worker("queued", engine, args.max_batch), engine, prompts, args.max_tokens))
        cont = asyncio.run(
            _throughput(_make_worker("continuous", engine, args.max_batch), engine, prompts, args.max_tokens)
        )
        speedup = (cont["tok_s"] / queued["tok_s"]) if queued["tok_s"] else None
        rows.append({"concurrency": n, "queued": queued, "continuous": cont, "speedup": speedup})
        print(
            f"n={n} queued_tok/s={queued['tok_s']:.3f} continuous_tok/s={cont['tok_s']:.3f} "
            f"speedup={speedup}"
        )

    payload = {
        "backend": backend,
        "max_tokens": args.max_tokens,
        "prompt_len": args.prompt_len,
        "max_batch": args.max_batch,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
