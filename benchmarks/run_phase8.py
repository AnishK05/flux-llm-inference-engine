#!/usr/bin/env python3
"""Run Phase 8 scenarios in-process (ASGI) or against a live URL."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Any

import httpx

from benchmarks.loadgen import LoadResult, result_to_json, run_load
from benchmarks.report import write_report
from benchmarks.scenarios import SCENARIOS, prompt_of_tokens
from flux.config import Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.naive_engine import NaiveEngine
from flux.runtime import apply_thread_caps, hardware_facts, hardware_line
from flux.server.app import create_app


def _expand(scenario, requests: int, mixed_cycle: bool) -> tuple[list[str], list[int]]:
    prompts: list[str] = []
    tokens: list[int] = []
    for i in range(requests):
        if scenario.mixed:
            short = i % 2 == 0
            p = scenario.prompt_tokens[0] if short else scenario.prompt_tokens[1]
            m = scenario.max_tokens[0] if short else scenario.max_tokens[1]
        else:
            p = int(scenario.prompt_tokens)
            m = int(scenario.max_tokens)
        prompts.append(prompt_of_tokens(int(p)))
        tokens.append(int(m))
    return prompts, tokens


async def _with_app(app, coro_factory):
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await coro_factory(client)


async def _run_one(client, scenario, engine_name, concurrency, requests, warmup, stream) -> LoadResult:
    prompts, tokens = _expand(scenario, requests, scenario.mixed)
    return await run_load(
        client,
        prompts=prompts,
        max_tokens=tokens,
        concurrency=concurrency,
        stream=stream,
        warmup=warmup,
        engine=engine_name,
        scenario=scenario.name,
        timeout=180.0,
    )


def _mean_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    keys = [
        "tok_s",
        "req_s",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "e2e_p99_ms",
    ]
    out = {key: runs[0].get(key) for key in ("scenario", "engine", "concurrency", "warmup")}
    agg: dict[str, Any] = {}
    for key in keys:
        values = [r["aggregates"].get(key) for r in runs if r["aggregates"].get(key) is not None]
        agg[key] = statistics.mean(values) if values else None
    agg["n_trials"] = len(runs)
    agg["statuses"] = runs[-1]["aggregates"].get("statuses")
    out["aggregates"] = agg
    out["trial"] = "mean"
    return out


def _story(mean_rows: list[dict[str, Any]]) -> dict[str, Any]:
    naive = next((r for r in mean_rows if r.get("engine") == "naive" and r.get("scenario") == "naive_vs_flux"), None)
    flux = next((r for r in mean_rows if r.get("engine") == "continuous" and r.get("scenario") == "naive_vs_flux"), None)
    queued = next((r for r in mean_rows if r.get("engine") == "queued"), None)
    cont = next((r for r in mean_rows if r.get("engine") == "continuous" and r.get("scenario") in {"mixed", "short_chat", "naive_vs_flux"}), None)
    lines = []
    payload: dict[str, Any] = {}
    if naive and flux:
        n_ttft = naive["aggregates"].get("ttft_p99_ms")
        f_ttft = flux["aggregates"].get("ttft_p99_ms")
        n_tok = naive["aggregates"].get("tok_s") or 0
        f_tok = flux["aggregates"].get("tok_s") or 0
        if n_ttft and f_ttft and n_ttft > 0:
            cut = (n_ttft - f_ttft) / n_ttft * 100.0
            payload["p99_ttft_cut_pct"] = cut
            if cut >= 0:
                lines.append(f"p99 TTFT naive {n_ttft:.1f} ms vs Flux {f_ttft:.1f} ms ({cut:.1f}% cut).")
            else:
                lines.append(
                    f"p99 TTFT naive {n_ttft:.1f} ms vs Flux {f_ttft:.1f} ms ({-cut:.1f}% higher on Flux)."
                )
        if n_tok and f_tok:
            payload["throughput_x"] = f_tok / n_tok
            lines.append(f"Aggregate tok/s Flux {f_tok:.2f} vs naive {n_tok:.2f} ({f_tok / n_tok:.2f}x).")
    if queued and cont and queued["aggregates"].get("tok_s") and cont["aggregates"].get("tok_s"):
        q, c = queued["aggregates"]["tok_s"], cont["aggregates"]["tok_s"]
        payload["queued_vs_continuous_x"] = c / q
        lines.append(f"Continuous vs queued tok/s {c:.2f} / {q:.2f} = {c / q:.2f}x.")
    return {"text": " ".join(lines), **payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--url", default="")
    parser.add_argument("--scenarios", default="naive_vs_flux,soak_200")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--out-dir", default="docs")
    parser.add_argument("--json-out", default="benchmarks/last_run.json")
    args = parser.parse_args()
    apply_thread_caps("auto")

    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    settings = Settings(load_model=False, max_new_tokens_cap=64)

    fake = {
        "naive": NaiveEngine(FakeLM(), FakeTokenizer()),
        "continuous": CachedEngine(FakeLM(), FakeTokenizer()),
        "queued": CachedEngine(FakeLM(), FakeTokenizer()),
    }
    engines: dict[str, Any] = dict(fake)
    if args.qwen and not args.url:
        from flux.engine.model_loader import load_causal_lm

        model, tokenizer = load_causal_lm(Settings(device="cpu", dtype="fp32"))
        engines["naive"] = NaiveEngine(model, tokenizer)
        engines["continuous"] = CachedEngine(model, tokenizer)
        engines["queued"] = CachedEngine(model, tokenizer)

    def app_for(name: str, *, control_plane: bool = False):
        pack = fake if control_plane else engines
        return create_app(
            settings=Settings(load_model=False, serve_engine=name, max_new_tokens_cap=64),
            engine=pack[name],
        )

    raw_runs: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []

    async def execute() -> None:
        for name in names:
            scenario = SCENARIOS[name]
            engine_names = ["continuous"]
            if name == "naive_vs_flux":
                engine_names = ["naive", "continuous"]
            elif name == "mixed":
                engine_names = ["queued", "continuous"]
            for engine_name in engine_names:
                for conc in scenario.concurrencies:
                    stream = engine_name != "naive"
                    trial_payloads = []
                    trials = args.trials if name == "naive_vs_flux" else max(1, min(args.trials, 3))
                    warmup = 0 if name == "soak_200" else args.warmup
                    requests = max(200, conc) if name == "soak_200" else args.requests
                    for trial in range(trials):
                        if args.url:
                            async with httpx.AsyncClient(base_url=args.url, timeout=180.0) as client:
                                result = await _run_one(
                                    client, scenario, engine_name, conc, requests, warmup, stream
                                )
                        else:
                            app = app_for(engine_name, control_plane=(name == "soak_200"))
                            result = await _with_app(
                                app,
                                lambda client, sc=scenario, en=engine_name, c=conc, rq=requests, w=warmup, st=stream: _run_one(
                                    client, sc, en, c, rq, w, st
                                ),
                            )
                        payload = result_to_json(result)
                        payload["trial"] = trial
                        trial_payloads.append(payload)
                        raw_runs.append(payload)
                        print(
                            f"{name} engine={engine_name} c={conc} trial={trial} "
                            f"tok/s={payload['aggregates']['tok_s']:.3f} "
                            f"p99_ttft={payload['aggregates']['ttft_p99_ms']} "
                            f"statuses={payload['aggregates']['statuses']}"
                        )
                    mean_rows.append(_mean_runs(trial_payloads))

    asyncio.run(execute())

    doc = {
        "hardware": hardware_facts(settings),
        "hardware_line": hardware_line(settings),
        "backend": "qwen" if args.qwen else ("remote" if args.url else "fake"),
        "warmup": args.warmup,
        "requests": args.requests,
        "trials": args.trials,
        "runs": mean_rows,
        "raw_trials": raw_runs,
        "note": hardware_facts(settings)["note"],
        "story": _story(mean_rows),
    }
    for run in raw_runs:
        run.pop("records", None)
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    report = write_report(doc, Path(args.out_dir))
    print(f"wrote {json_path} and {report}")


if __name__ == "__main__":
    main()
