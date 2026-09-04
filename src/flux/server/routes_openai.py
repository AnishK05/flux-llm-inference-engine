from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from flux.config import SERVED_MODEL_ALIASES, SERVED_MODEL_ID
from flux.engine.types import SamplingParams
from flux.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _engine(request: Request) -> Any:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return engine


def _sampling(request: Request, temperature: float, top_p: float, max_tokens: int) -> SamplingParams:
    settings = request.app.state.settings
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=min(max_tokens, settings.max_new_tokens_cap),
    )


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(body: CompletionRequest, request: Request) -> CompletionResponse:
    if body.model not in SERVED_MODEL_ALIASES:
        raise HTTPException(status_code=404, detail=f"unknown model {body.model!r}")

    engine = _engine(request)
    sampling = _sampling(request, body.temperature, body.top_p, body.max_tokens)
    lock = request.app.state.generate_lock
    async with lock:
        result = await asyncio.to_thread(engine.generate, body.prompt, sampling)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=SERVED_MODEL_ID,
        choices=[CompletionChoice(text=result.text, finish_reason=result.finish_reason)],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    body: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse:
    if body.model not in SERVED_MODEL_ALIASES:
        raise HTTPException(status_code=404, detail=f"unknown model {body.model!r}")

    engine = _engine(request)
    sampling = _sampling(request, body.temperature, body.top_p, body.max_tokens)
    messages = [item.model_dump() for item in body.messages]
    lock = request.app.state.generate_lock
    async with lock:
        result = await asyncio.to_thread(engine.generate_chat, messages, sampling)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=SERVED_MODEL_ID,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )
