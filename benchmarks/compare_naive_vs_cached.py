#!/usr/bin/env python3
"""Compare naive full-recompute vs KV-cached decode.

Default: FakeLM (CI-safe; timings are not meaningful).
Pass --qwen to load Qwen2.5-0.5B-Instruct and write a real TTFT/TPOT table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.naive_engine import NaiveEngine
from flux.engine.types import SamplingParams
from flux.runtime import apply_thread_caps


def _run(engine, prompt: str, max_tokens: int):
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    started = time.perf_counter()
    result = engine.generate(prompt, params)
    wall = time.perf_counter() - started
    return {
        "engine": result.engine,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_s": result.latency_s,
        "ttft_s": result.ttft_s,
        "tpot_s": result.tpot_s,
        "wall_s": wall,
        "output": result.text,
    }


def _prompt_of_len(target_tokens: int) -> str:
    # Roughly one token per word for the FakeLM / English Qwen tokenizer.
    word = "alpha "
    return (word * target_tokens).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--lengths", default="32,128")
    parser.add_argument("--out", default="docs/phase2_naive_vs_cached.json")
    args = parser.parse_args()
    apply_thread_caps("auto")

    if args.qwen:
        from flux.config import Settings
        from flux.engine.model_loader import load_causal_lm

        settings = Settings(device="cpu", dtype="fp32")
        model, tokenizer = load_causal_lm(settings)
        naive = NaiveEngine(model, tokenizer, device="cpu")
        cached = CachedEngine(model, tokenizer, device="cpu")
        backend = "qwen"
    else:
        naive = NaiveEngine(FakeLM(), FakeTokenizer(), device="cpu")
        cached = CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")
        backend = "fake"

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    rows = []
    for length in lengths:
        prompt = _prompt_of_len(length)
        naive_row = _run(naive, prompt, args.max_tokens)
        cached_row = _run(cached, prompt, args.max_tokens)
        rows.append({"target_prompt_len": length, "naive": naive_row, "cached": cached_row})
        print(
            f"len~{length} naive_tpot={naive_row['tpot_s']} cached_tpot={cached_row['tpot_s']} "
            f"naive_ttft={naive_row['ttft_s']:.4f} cached_ttft={cached_row['ttft_s']:.4f}"
        )

    payload = {"backend": backend, "max_tokens": args.max_tokens, "rows": rows}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
