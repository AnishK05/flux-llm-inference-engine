from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from flux.config import SERVED_MODEL_ALIASES, SERVED_MODEL_ID
from flux.engine.scheduler import QueueFull
from flux.engine.sequence import Sequence, SequenceStatus
from flux.engine.serving import WORKER_ENGINES, normalize_serve_engine
from flux.engine.tokenizer import encode_chat, encode_text
from flux.engine.types import GenerateResult, SamplingParams
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


def _fit_max_seq(request: Request, input_ids: Any, sampling: SamplingParams) -> SamplingParams:
    settings = request.app.state.settings
    prompt_len = int(input_ids.shape[-1])
    if prompt_len > settings.max_seq_len:
        raise HTTPException(
            status_code=400,
            detail=f"prompt length {prompt_len} exceeds max_seq_len {settings.max_seq_len}",
        )
    room = settings.max_seq_len - prompt_len
    if room < 1:
        raise HTTPException(status_code=400, detail="prompt already at max_seq_len")
    if sampling.max_tokens > room:
        return SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=room,
            stop_token_ids=sampling.stop_token_ids,
            ignore_eos=sampling.ignore_eos,
        )
    return sampling


def _uses_worker(request: Request) -> bool:
    mode = normalize_serve_engine(request.app.state.settings.serve_engine)
    return mode in WORKER_ENGINES and getattr(request.app.state, "scheduler", None) is not None


async def _enqueue(request: Request, prompt_ids: Any, sampling: SamplingParams) -> GenerateResult:
    scheduler = request.app.state.scheduler
    seq = Sequence(prompt_ids=prompt_ids, sampling=sampling)
    try:
        scheduler.submit(seq)
    except QueueFull as exc:
        raise HTTPException(
            status_code=429,
            detail="waiting queue is full",
            headers={"Retry-After": "1"},
        ) from exc
    await seq.finished_event.wait()
    if seq.error is not None:
        raise HTTPException(status_code=500, detail=str(seq.error))
    if seq.status == SequenceStatus.ABORTED:
        raise HTTPException(status_code=503, detail="server shutting down")
    if seq.result is None:
        raise HTTPException(status_code=500, detail="sequence finished without a result")
    return seq.result


async def _generate_ids(request: Request, engine: Any, prompt_ids: Any, sampling: SamplingParams) -> GenerateResult:
    sampling = _fit_max_seq(request, prompt_ids, sampling)
    if _uses_worker(request):
        return await _enqueue(request, prompt_ids, sampling)
    lock = request.app.state.generate_lock
    async with lock:
        return await asyncio.to_thread(engine.generate_ids, prompt_ids, sampling)


def _usage(result: GenerateResult) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
    )


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(body: CompletionRequest, request: Request) -> CompletionResponse:
    if body.model not in SERVED_MODEL_ALIASES:
        raise HTTPException(status_code=404, detail=f"unknown model {body.model!r}")

    engine = _engine(request)
    sampling = _sampling(request, body.temperature, body.top_p, body.max_tokens)
    prompt_ids = encode_text(engine.tokenizer, body.prompt, device=getattr(engine, "device", "cpu"))
    result = await _generate_ids(request, engine, prompt_ids, sampling)

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=SERVED_MODEL_ID,
        choices=[CompletionChoice(text=result.text, finish_reason=result.finish_reason)],
        usage=_usage(result),
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
    prompt_ids = encode_chat(engine.tokenizer, messages, device=getattr(engine, "device", "cpu"))
    result = await _generate_ids(request, engine, prompt_ids, sampling)

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
        usage=_usage(result),
    )
