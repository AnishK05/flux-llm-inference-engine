#!/usr/bin/env python3
"""Closed-loop HTTP loadgen for Flux.

N workers each send the next request when the previous one finishes.
Open-loop Poisson is out of scope.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from flux.config import SERVED_MODEL_ID


@dataclass
class RequestRecord:
    http_status: int
    e2e_ms: float
    ttft_ms: float | None
    engine_ttft_ms: float | None
    tpot_ms: float | None
    prompt_tokens: int
    output_tokens: int
    error: str = ""


@dataclass
class LoadResult:
    scenario: str
    engine: str
    concurrency: int
    warmup: int
    records: list[RequestRecord] = field(default_factory=list)
    wall_s: float = 0.0
    statuses: dict[str, int] = field(default_factory=dict)

    def measured(self) -> list[RequestRecord]:
        return [row for row in self.records[self.warmup :] if row.http_status == 200]

    def aggregates(self) -> dict[str, Any]:
        rows = self.measured()
        wall = self.wall_s
        tokens = sum(row.output_tokens for row in rows)
        ok = [row for row in rows if row.e2e_ms > 0]
        ttfts = [row.engine_ttft_ms or row.ttft_ms or 0.0 for row in ok if (row.engine_ttft_ms or row.ttft_ms)]
        e2es = [row.e2e_ms for row in ok]
        return {
            "n_measured": len(rows),
            "n_ok": len(ok),
            "output_tokens": tokens,
            "tok_s": (tokens / wall) if wall > 0 else 0.0,
            "req_s": (len(ok) / wall) if wall > 0 else 0.0,
            "ttft_p50_ms": _pct(ttfts, 50),
            "ttft_p95_ms": _pct(ttfts, 95),
            "ttft_p99_ms": _pct(ttfts, 99),
            "e2e_p50_ms": _pct(e2es, 50),
            "e2e_p95_ms": _pct(e2es, 95),
            "e2e_p99_ms": _pct(e2es, 99),
            "statuses": dict(self.statuses),
            "wall_s": wall,
        }


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(round((q / 100.0) * (len(ordered) - 1)))
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return {"done": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def _one_stream(client: httpx.AsyncClient, url: str, body: dict, timeout: float) -> RequestRecord:
    started = time.perf_counter()
    http_ttft = None
    engine_ttft = None
    tpot = None
    prompt_tokens = 0
    output_tokens = 0
    status = 0
    text_chunks = 0
    try:
        async with client.stream("POST", url, json=body, timeout=timeout) as response:
            status = response.status_code
            async for line in response.aiter_lines():
                event = _parse_sse_line(line)
                if not event or event.get("done"):
                    continue
                choices = event.get("choices") or [{}]
                text = (choices[0] or {}).get("text") or (choices[0] or {}).get("delta", {}).get("content")
                if text and http_ttft is None:
                    http_ttft = (time.perf_counter() - started) * 1000.0
                if text:
                    text_chunks += 1
                flux = event.get("flux") or {}
                if "ttft_ms" in flux:
                    engine_ttft = float(flux["ttft_ms"])
                if "tpot_ms" in flux:
                    tpot = float(flux["tpot_ms"])
                usage = event.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    output_tokens = int(usage.get("completion_tokens") or 0)
        e2e = (time.perf_counter() - started) * 1000.0
        if output_tokens <= 0:
            output_tokens = text_chunks
        if tpot is None and output_tokens > 1 and http_ttft is not None:
            tpot = (e2e - http_ttft) / (output_tokens - 1)
        return RequestRecord(
            http_status=status,
            e2e_ms=e2e,
            ttft_ms=http_ttft,
            engine_ttft_ms=engine_ttft,
            tpot_ms=tpot,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        return RequestRecord(
            http_status=status or 0,
            e2e_ms=(time.perf_counter() - started) * 1000.0,
            ttft_ms=http_ttft,
            engine_ttft_ms=engine_ttft,
            tpot_ms=tpot,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            error=str(exc),
        )


async def _one_json(client: httpx.AsyncClient, url: str, body: dict, timeout: float) -> RequestRecord:
    started = time.perf_counter()
    try:
        response = await client.post(url, json=body, timeout=timeout)
        e2e = (time.perf_counter() - started) * 1000.0
        payload = {}
        try:
            payload = response.json()
        except Exception:
            payload = {}
        usage = payload.get("usage") or {}
        engine_ttft = response.headers.get("x-flux-ttft-ms")
        tpot = response.headers.get("x-flux-tpot-ms")
        return RequestRecord(
            http_status=response.status_code,
            e2e_ms=e2e,
            ttft_ms=float(engine_ttft) if engine_ttft else None,
            engine_ttft_ms=float(engine_ttft) if engine_ttft else None,
            tpot_ms=float(tpot) if tpot else None,
            prompt_tokens=int(usage.get("prompt_tokens") or response.headers.get("x-flux-prompt-tokens") or 0),
            output_tokens=int(
                usage.get("completion_tokens") or response.headers.get("x-flux-completion-tokens") or 0
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return RequestRecord(
            http_status=0,
            e2e_ms=(time.perf_counter() - started) * 1000.0,
            ttft_ms=None,
            engine_ttft_ms=None,
            tpot_ms=None,
            prompt_tokens=0,
            output_tokens=0,
            error=str(exc),
        )


async def run_load(
    client: httpx.AsyncClient,
    *,
    prompts: list[str],
    max_tokens: list[int],
    concurrency: int,
    stream: bool = True,
    timeout: float = 120.0,
    warmup: int = 0,
    engine: str = "",
    scenario: str = "",
) -> LoadResult:
    n = len(prompts)
    url = "/v1/completions"
    next_i = 0
    lock = asyncio.Lock()
    records: list[RequestRecord] = []

    async def worker() -> None:
        nonlocal next_i
        while True:
            async with lock:
                idx = next_i
                next_i += 1
            if idx >= n:
                return
            body = {
                "model": SERVED_MODEL_ID,
                "prompt": prompts[idx],
                "max_tokens": max_tokens[idx],
                "temperature": 0,
                "stream": stream,
            }
            if stream:
                rec = await _one_stream(client, url, body, timeout)
            else:
                rec = await _one_json(client, url, body, timeout)
            records.append(rec)

    started = time.perf_counter()
    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(max(1, concurrency))])
    wall = time.perf_counter() - started
    statuses: dict[str, int] = {}
    for rec in records:
        key = str(rec.http_status)
        statuses[key] = statuses.get(key, 0) + 1
    return LoadResult(
        scenario=scenario,
        engine=engine,
        concurrency=concurrency,
        warmup=min(warmup, len(records)),
        records=records,
        wall_s=wall,
        statuses=statuses,
    )


def result_to_json(result: LoadResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["aggregates"] = result.aggregates()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop Flux loadgen")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    stream = not args.no_stream

    async def _run() -> LoadResult:
        async with httpx.AsyncClient(base_url=args.url, timeout=120.0) as client:
            return await run_load(
                client,
                prompts=[args.prompt] * args.requests,
                max_tokens=[args.max_tokens] * args.requests,
                concurrency=args.concurrency,
                stream=stream,
                warmup=args.warmup,
                scenario="adhoc",
            )

    result = asyncio.run(_run())
    payload = result_to_json(result)
    print(json.dumps(payload["aggregates"], indent=2))
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
