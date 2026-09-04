from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from flux.config import SERVED_MODEL_ALIASES, SERVED_MODEL_ID
from flux.engine.naive_engine import NaiveEngine
from flux.engine.types import SamplingParams
from flux.server.schemas import CompletionChoice, CompletionRequest, CompletionResponse, CompletionUsage

logger = logging.getLogger(__name__)
router = APIRouter()


def _engine(request: Request) -> NaiveEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return engine


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(body: CompletionRequest, request: Request) -> CompletionResponse:
    if body.model not in SERVED_MODEL_ALIASES:
        raise HTTPException(status_code=404, detail=f"unknown model {body.model!r}")

    settings = request.app.state.settings
    max_tokens = min(body.max_tokens, settings.max_new_tokens_cap)
    engine = _engine(request)
    sampling = SamplingParams(temperature=body.temperature, top_p=body.top_p, max_tokens=max_tokens)

    lock = request.app.state.generate_lock
    async with lock:
        result = await _run_generate(engine, body.prompt, sampling)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=SERVED_MODEL_ID,
        choices=[
            CompletionChoice(text=result.text, finish_reason=result.finish_reason),
        ],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


async def _run_generate(engine: NaiveEngine, prompt: str, sampling: SamplingParams):
    return await asyncio.to_thread(engine.generate, prompt, sampling)
