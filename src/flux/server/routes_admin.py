from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from flux.engine.serving import normalize_serve_engine
from flux.metrics.prometheus import RECENT_TTFT, metrics_response
from flux.runtime import probe, rss_bytes

router = APIRouter()

_BENCH_CANDIDATES = (
    Path("benchmarks/last_run.json"),
    Path("/app/benchmarks/last_run.json"),
    Path(__file__).resolve().parents[3] / "benchmarks" / "last_run.json",
)


@router.get("/health")
def health(request: Request) -> dict:
    engine = getattr(request.app.state, "engine", None)
    loaded = engine is not None and getattr(engine, "model_loaded", False)
    payload = probe(request.app.state.settings, model_loaded=loaded)
    worker = getattr(request.app.state, "worker", None)
    if worker is not None:
        payload["engine_name"] = worker.engine_name
    elif engine is not None:
        payload["engine_name"] = getattr(engine, "engine_name", None)
    payload["serve_engine"] = normalize_serve_engine(request.app.state.settings.serve_engine)
    control = getattr(request.app.state, "control", None)
    payload["redis"] = bool(control is not None and getattr(control, "enabled", False))
    return payload


@router.get("/ready")
def ready(request: Request):
    engine = getattr(request.app.state, "engine", None)
    loaded = engine is not None and getattr(engine, "model_loaded", False)
    if not loaded:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "model not loaded"})
    return {"ready": True, "rss_bytes": rss_bytes()}


@router.get("/admin/stats")
def admin_stats(request: Request) -> dict:
    settings = request.app.state.settings
    scheduler = getattr(request.app.state, "scheduler", None)
    worker = getattr(request.app.state, "worker", None)
    waiting_ids = scheduler.queue.snapshot_ids() if scheduler is not None else []
    running = list(worker.stats.running) if worker is not None else []
    running_ids = [seq.id for seq in running]
    last_batch = worker.stats.last_batch_size if worker is not None else 0
    tokens = worker.stats.tokens_generated if worker is not None else 0
    pool = getattr(request.app.state, "block_pool", None)
    if pool is None and scheduler is not None:
        pool = getattr(scheduler, "pool", None)
    kv = pool.snapshot() if pool is not None else {
        "kv_blocks_used": 0,
        "kv_blocks_free": 0,
        "kv_blocks_total": 0,
        "kv_block_size": settings.block_size,
    }
    rate = getattr(request.app.state, "token_rate", None)
    tok_s = rate.observe(tokens) if rate is not None else 0.0
    payload = {
        "waiting": len(waiting_ids),
        "running": len(running_ids),
        "in_flight": len(waiting_ids) + len(running_ids),
        "max_waiting": settings.max_waiting,
        "max_batch_size": settings.max_batch_size,
        "last_batch_size": last_batch,
        "tokens_generated": tokens,
        "tok_s": round(tok_s, 3),
        "ttft_p50_ms": RECENT_TTFT.p50_ms(),
        "rss_bytes": rss_bytes(),
        "waiting_ids": waiting_ids,
        "running_ids": running_ids,
        "serve_engine": normalize_serve_engine(settings.serve_engine),
        "scheduler": getattr(scheduler, "policy", settings.scheduler) if scheduler else settings.scheduler,
    }
    payload.update(kv)
    prefix = getattr(request.app.state, "prefix_cache", None)
    if prefix is None and scheduler is not None:
        prefix = getattr(scheduler, "prefix_cache", None)
    if prefix is not None:
        payload.update(prefix.snapshot())
    else:
        payload.update(
            {
                "prefix_entries": 0,
                "prefix_hits": 0,
                "prefix_misses": 0,
                "prefix_tokens_saved": 0,
                "prefix_live_refs": 0,
            }
        )
    return payload


@router.post("/admin/abort/{request_id}")
def admin_abort(request_id: str, request: Request) -> dict:
    seq = _find_sequence(request, request_id)
    if seq is None:
        raise HTTPException(status_code=404, detail=f"unknown request {request_id}")
    scheduler = request.app.state.scheduler
    if scheduler is not None and scheduler.queue.remove(seq):
        from flux.engine.worker import _abort_one

        _abort_one(seq, scheduler)
        return {"aborted": request_id, "was": "waiting"}
    seq.request_abort()
    return {"aborted": request_id, "was": "running"}


@router.get("/admin/bench")
def admin_bench() -> dict:
    for path in _BENCH_CANDIDATES:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="benchmarks/last_run.json not found")


@router.get("/v1/requests/{request_id}")
def get_request(request_id: str, request: Request) -> dict:
    control = getattr(request.app.state, "control", None)
    if control is None:
        raise HTTPException(status_code=404, detail="unknown request")
    job = control.get_job(request_id)
    if job is None:
        seq = _find_sequence(request, request_id)
        if seq is None:
            raise HTTPException(status_code=404, detail="unknown request")
        return {
            "id": seq.id,
            "status": seq.status.value,
            "output_tokens": len(seq.output_ids),
        }
    return job


@router.get("/metrics")
def metrics(request: Request):
    return metrics_response(request.app)


def _find_sequence(request: Request, request_id: str):
    scheduler = getattr(request.app.state, "scheduler", None)
    worker = getattr(request.app.state, "worker", None)
    if scheduler is not None:
        for seq in scheduler.queue:
            if seq.id == request_id:
                return seq
    if worker is not None:
        for seq in worker.stats.running:
            if seq.id == request_id:
                return seq
    return None
