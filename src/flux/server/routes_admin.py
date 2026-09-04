from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from flux.runtime import probe, rss_bytes

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    engine = getattr(request.app.state, "engine", None)
    loaded = engine is not None and getattr(engine, "model_loaded", False)
    payload = probe(request.app.state.settings, model_loaded=loaded)
    if engine is not None:
        payload["engine_name"] = getattr(engine, "engine_name", None)
    return payload


@router.get("/ready")
def ready(request: Request):
    engine = getattr(request.app.state, "engine", None)
    loaded = engine is not None and getattr(engine, "model_loaded", False)
    if not loaded:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "model not loaded"})
    return {"ready": True, "rss_bytes": rss_bytes()}
