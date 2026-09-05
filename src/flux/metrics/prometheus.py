"""Prometheus metrics for the Flux serving loop."""

from __future__ import annotations

from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, REGISTRY, generate_latest
from starlette.responses import Response

from flux.control import RecentTtft
from flux.engine.sequence import SequenceStatus
from flux.runtime import rss_bytes

_DONE = {SequenceStatus.FINISHED, SequenceStatus.ABORTED, SequenceStatus.ERROR}

RECENT_TTFT = RecentTtft()


def _existing(name: str):
    collectors = REGISTRY._names_to_collectors
    found = collectors.get(name)
    if found is not None:
        return found
    # Histograms register as {name}_bucket/_count/_sum; counters as {name}_total.
    for suffix in ("_bucket", "_count", "_sum", "_total", "_created"):
        found = collectors.get(name + suffix)
        if found is not None:
            return found
    return None


def _gauge(name: str, documentation: str) -> Gauge:
    found = _existing(name)
    if isinstance(found, Gauge):
        return found
    return Gauge(name, documentation)


def _counter(name: str, documentation: str, labels: list[str] | None = None) -> Counter:
    found = _existing(name)
    if isinstance(found, Counter):
        return found
    if labels:
        return Counter(name, documentation, labels)
    return Counter(name, documentation)


def _histogram(name: str, documentation: str, buckets: tuple[float, ...]) -> Histogram:
    found = _existing(name)
    if isinstance(found, Histogram):
        return found
    return Histogram(name, documentation, buckets=buckets)


TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0, 12.0)
STEP_BUCKETS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0)
E2E_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

WAITING = _gauge("flux_waiting", "Sequences waiting in the admission queue")
RUNNING = _gauge("flux_running", "Sequences in prefill or decode")
IN_FLIGHT = _gauge("flux_in_flight", "Waiting plus running sequences")
KV_USED = _gauge("flux_kv_blocks_used", "KV blocks currently reserved")
KV_TOTAL = _gauge("flux_kv_blocks_total", "KV blocks in the accounting pool")
DECODE_BATCH = _gauge("flux_decode_batch_size", "Last decode batch size")
PREFIX_ENTRIES = _gauge("flux_prefix_entries", "Stored prefix-cache entries")
PREFIX_HITS = _gauge("flux_prefix_hits", "Prefix-cache hits since process start")
PREFIX_MISSES = _gauge("flux_prefix_misses", "Prefix-cache misses since process start")
PREFIX_TOKENS = _gauge("flux_prefix_tokens_saved", "Prompt tokens skipped by prefix reuse")
RSS = _gauge("flux_rss_bytes", "Process resident set size in bytes")
TOKENS = _counter("flux_output_tokens_total", "Output tokens generated")
REQUESTS = _counter("flux_http_requests_total", "HTTP generate requests", ["endpoint", "status"])
TTFT = _histogram("flux_ttft_seconds", "Engine time to first token", TTFT_BUCKETS)
E2E = _histogram("flux_e2e_seconds", "Engine end-to-end generate latency", E2E_BUCKETS)
PREFILL = _histogram("flux_prefill_seconds", "Prefill forward time", STEP_BUCKETS)
DECODE = _histogram("flux_decode_step_seconds", "One batched decode step", STEP_BUCKETS)


def update_gauges(app: Any) -> None:
    scheduler = getattr(app.state, "scheduler", None)
    worker = getattr(app.state, "worker", None)
    pool = getattr(app.state, "block_pool", None)
    if pool is None and scheduler is not None:
        pool = getattr(scheduler, "pool", None)
    waiting = len(scheduler.queue) if scheduler is not None else 0
    running = (
        sum(1 for seq in worker.stats.running if seq.status not in _DONE)
        if worker is not None
        else 0
    )
    WAITING.set(waiting)
    RUNNING.set(running)
    IN_FLIGHT.set(waiting + running)
    DECODE_BATCH.set(worker.stats.last_batch_size if worker is not None else 0)
    if pool is not None:
        KV_USED.set(pool.used_blocks)
        KV_TOTAL.set(pool.num_blocks)
    else:
        KV_USED.set(0)
        KV_TOTAL.set(0)
    prefix = getattr(app.state, "prefix_cache", None)
    if prefix is None and scheduler is not None:
        prefix = getattr(scheduler, "prefix_cache", None)
    if prefix is not None:
        snap = prefix.snapshot()
        PREFIX_ENTRIES.set(snap["prefix_entries"])
        PREFIX_HITS.set(snap["prefix_hits"])
        PREFIX_MISSES.set(snap["prefix_misses"])
        PREFIX_TOKENS.set(snap["prefix_tokens_saved"])
    else:
        PREFIX_ENTRIES.set(0)
        PREFIX_HITS.set(0)
        PREFIX_MISSES.set(0)
        PREFIX_TOKENS.set(0)
    RSS.set(rss_bytes())


def observe_step(kind: str, seconds: float) -> None:
    if seconds < 0:
        return
    if kind == "prefill":
        PREFILL.observe(seconds)
    elif kind == "decode":
        DECODE.observe(seconds)


def observe_finished(result: Any) -> None:
    if result is None:
        return
    ttft = float(getattr(result, "ttft_s", 0.0) or 0.0)
    e2e = float(getattr(result, "latency_s", 0.0) or 0.0)
    tokens = int(getattr(result, "completion_tokens", 0) or 0)
    if ttft > 0:
        TTFT.observe(ttft)
        RECENT_TTFT.add(ttft)
    if e2e > 0:
        E2E.observe(e2e)
    if tokens > 0:
        TOKENS.inc(tokens)


def observe_request(endpoint: str, status: int) -> None:
    REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()


def metrics_response(app: Any) -> Response:
    update_gauges(app)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
