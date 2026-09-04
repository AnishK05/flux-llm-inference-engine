from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from flux.config import Settings, get_settings
from flux.engine.naive_engine import NaiveEngine
from flux.runtime import apply_thread_caps
from flux.server.routes_admin import router as admin_router
from flux.server.routes_openai import router as openai_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None, engine: NaiveEngine | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        apply_thread_caps(settings.intra_threads)
        if app.state.engine is None and settings.load_model:
            logger.info("startup: loading %s", settings.model)
            from flux.engine.model_loader import load_causal_lm

            model, tokenizer = await asyncio.to_thread(load_causal_lm, settings)
            app.state.engine = NaiveEngine(model, tokenizer, device=settings.device)
            logger.info("startup: model ready")
        yield

    app = FastAPI(
        title="Flux",
        version="0.1.0",
        description="CPU LLM inference server — Phase 1 naive baseline",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.generate_lock = asyncio.Lock()
    app.include_router(admin_router)
    app.include_router(openai_router)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request, exc: RequestValidationError) -> JSONResponse:
        details = []
        for err in exc.errors():
            item = dict(err)
            ctx = item.get("ctx")
            if ctx:
                item["ctx"] = {key: str(value) for key, value in ctx.items()}
            details.append(item)
        return JSONResponse(status_code=400, content={"detail": details})

    return app


def get_app() -> FastAPI:
    return create_app()


# Uvicorn target: flux.server.app:app
app = get_app()


def main() -> None:
    import uvicorn

    uvicorn.run("flux.server.app:app", host="0.0.0.0", port=8000, reload=False)
