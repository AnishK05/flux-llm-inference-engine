"""Attach a Phase 4/5 worker to a FastAPI app."""

from __future__ import annotations

import logging
from typing import Any

from flux.config import Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.scheduler import RequestQueue, Scheduler
from flux.engine.worker import ContinuousWorker, QueuedWorker

logger = logging.getLogger(__name__)

WORKER_ENGINES = frozenset({"queued", "continuous"})


def normalize_serve_engine(name: str | None) -> str:
    value = (name or "continuous").strip().lower()
    if value in {"naive", "cached", "queued", "continuous"}:
        return value
    logger.warning("unknown FLUX_SERVE_ENGINE=%r, falling back to continuous", name)
    return "continuous"


def attach_worker(app: Any, settings: Settings, engine: Any) -> QueuedWorker | ContinuousWorker | None:
    """Create scheduler + worker when serving through the request queue."""
    mode = normalize_serve_engine(settings.serve_engine)
    app.state.scheduler = None
    app.state.worker = None
    if mode not in WORKER_ENGINES:
        return None
    if engine is None:
        return None
    if not isinstance(engine, CachedEngine):
        logger.warning("serve_engine=%s requires CachedEngine; not starting a worker", mode)
        return None
    queue = RequestQueue(settings.max_waiting)
    scheduler = Scheduler(queue, max_batch=settings.max_batch_size)
    worker: QueuedWorker | ContinuousWorker
    if mode == "queued":
        worker = QueuedWorker(engine, scheduler)
    else:
        worker = ContinuousWorker(engine, scheduler)
    app.state.scheduler = scheduler
    app.state.worker = worker
    logger.info(
        "worker attached mode=%s max_waiting=%d max_batch=%d",
        mode,
        settings.max_waiting,
        settings.max_batch_size,
    )
    return worker
