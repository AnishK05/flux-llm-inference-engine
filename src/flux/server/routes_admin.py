from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from flux.engine.serving import normalize_serve_engine
from flux.metrics.prometheus import metrics_response
from flux.runtime import probe, rss_bytes

router = APIRouter()


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
    waiting = len(scheduler.queue) if scheduler is not None else 0
    running = len(worker.stats.running) if worker is not None else 0
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
    payload = {
        "waiting": waiting,
        "running": running,
        "in_flight": waiting + running,
        "max_waiting": settings.max_waiting,
        "max_batch_size": settings.max_batch_size,
        "last_batch_size": last_batch,
        "tokens_generated": tokens,
        "serve_engine": normalize_serve_engine(settings.serve_engine),
        "scheduler": getattr(scheduler, "policy", settings.scheduler) if scheduler else settings.scheduler,
    }
    payload.update(kv)
    return payload


@router.get("/metrics")
def metrics(request: Request):
    return metrics_response(request.app)
