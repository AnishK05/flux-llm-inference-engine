from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from flux.config import SERVED_MODEL_ALIASES, SERVED_MODEL_ID
from flux.engine.scheduler import QueueFull, RequestTooLarge
from flux.engine.sequence import Sequence, SequenceStatus
from flux.engine.serving import WORKER_ENGINES, normalize_serve_engine
from flux.engine.tokenizer import decode_delta, encode_chat, encode_text
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
from flux.server.sse import chat_chunk, completion_chunk, iter_text_deltas, sse_data

logger = logging.getLogger(__name__)
router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


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


def _submit(request: Request, prompt_ids: Any, sampling: SamplingParams) -> Sequence:
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
    except RequestTooLarge as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return seq


async def _wait_result(seq: Sequence) -> GenerateResult:
    await seq.finished_event.wait()
    if seq.error is not None:
        raise HTTPException(status_code=500, detail=str(seq.error))
    if seq.status == SequenceStatus.ABORTED:
        raise HTTPException(status_code=499, detail="request aborted")
    if seq.result is None:
        raise HTTPException(status_code=500, detail="sequence finished without a result")
    return seq.result


async def _generate_ids(request: Request, engine: Any, prompt_ids: Any, sampling: SamplingParams) -> GenerateResult:
    sampling = _fit_max_seq(request, prompt_ids, sampling)
    if _uses_worker(request):
        seq = _submit(request, prompt_ids, sampling)
        return await _wait_result(seq)
    lock = request.app.state.generate_lock
    async with lock:
        return await asyncio.to_thread(engine.generate_ids, prompt_ids, sampling)


def _usage(result: GenerateResult) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.prompt_tokens + result.completion_tokens,
    )


def _sse_response(iterator: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(iterator, media_type="text/event-stream", headers=_SSE_HEADERS)


async def _stream_completion(
    request: Request,
    seq: Sequence,
    tokenizer: Any,
    request_id: str,
    created: int,
) -> AsyncIterator[bytes]:
    try:
        async for delta in iter_text_deltas(seq, tokenizer):
            if await request.is_disconnected():
                seq.request_abort()
                return
            yield sse_data(completion_chunk(request_id, created, delta))
        reason = seq.finish_reason or ("abort" if seq.abort_requested else "length")
        if seq.status != SequenceStatus.ABORTED:
            yield sse_data(completion_chunk(request_id, created, "", finish_reason=reason))
        yield sse_data("[DONE]")
    except asyncio.CancelledError:
        seq.request_abort()
        raise
    finally:
        if not seq.finished_event.is_set():
            seq.request_abort()


async def _stream_chat(
    request: Request,
    seq: Sequence,
    tokenizer: Any,
    request_id: str,
    created: int,
) -> AsyncIterator[bytes]:
    try:
        yield sse_data(chat_chunk(request_id, created, {"role": "assistant"}))
        async for delta in iter_text_deltas(seq, tokenizer):
            if await request.is_disconnected():
                seq.request_abort()
                return
            yield sse_data(chat_chunk(request_id, created, {"content": delta}))
        reason = seq.finish_reason or ("abort" if seq.abort_requested else "length")
        if seq.status != SequenceStatus.ABORTED:
            yield sse_data(chat_chunk(request_id, created, {}, finish_reason=reason))
        yield sse_data("[DONE]")
    except asyncio.CancelledError:
        seq.request_abort()
        raise
    finally:
        if not seq.finished_event.is_set():
            seq.request_abort()


async def _stream_from_result(
    result: GenerateResult,
    kind: str,
    request_id: str,
    created: int,
    tokenizer: Any,
) -> AsyncIterator[bytes]:
    prev = ""
    if kind == "chat":
        yield sse_data(chat_chunk(request_id, created, {"role": "assistant"}))
    ids: list[int] = []
    for token in result.output_token_ids:
        ids.append(token)
        prev, delta = decode_delta(tokenizer, ids, prev)
        if not delta:
            continue
        if kind == "chat":
            yield sse_data(chat_chunk(request_id, created, {"content": delta}))
        else:
            yield sse_data(completion_chunk(request_id, created, delta))
    if kind == "chat":
        yield sse_data(chat_chunk(request_id, created, {}, finish_reason=result.finish_reason))
    else:
        yield sse_data(completion_chunk(request_id, created, "", finish_reason=result.finish_reason))
    yield sse_data("[DONE]")


@router.post("/v1/completions")
async def create_completion(body: CompletionRequest, request: Request):
    if body.model not in SERVED_MODEL_ALIASES:
        raise HTTPException(status_code=404, detail=f"unknown model {body.model!r}")

    engine = _engine(request)
    sampling = _sampling(request, body.temperature, body.top_p, body.max_tokens)
    prompt_ids = encode_text(engine.tokenizer, body.prompt, device=getattr(engine, "device", "cpu"))
    sampling = _fit_max_seq(request, prompt_ids, sampling)
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"

    if body.stream:
        if _uses_worker(request):
            seq = _submit(request, prompt_ids, sampling)
            return _sse_response(_stream_completion(request, seq, engine.tokenizer, request_id, created))
        result = await _generate_ids(request, engine, prompt_ids, sampling)
        return _sse_response(_stream_from_result(result, "completion", request_id, created, engine.tokenizer))

    result = await _generate_ids(request, engine, prompt_ids, sampling)
    return CompletionResponse(
        id=request_id,
        created=created,
        model=SERVED_MODEL_ID,
        choices=[CompletionChoice(text=result.text, finish_reason=result.finish_reason)],
        usage=_usage(result),
    )


@router.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    if body.model not in SERVED_MODEL_ALIASES:
        raise HTTPException(status_code=404, detail=f"unknown model {body.model!r}")

    engine = _engine(request)
    sampling = _sampling(request, body.temperature, body.top_p, body.max_tokens)
    messages = [item.model_dump() for item in body.messages]
    prompt_ids = encode_chat(engine.tokenizer, messages, device=getattr(engine, "device", "cpu"))
    sampling = _fit_max_seq(request, prompt_ids, sampling)
    created = int(time.time())
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if body.stream:
        if _uses_worker(request):
            seq = _submit(request, prompt_ids, sampling)
            return _sse_response(_stream_chat(request, seq, engine.tokenizer, request_id, created))
        result = await _generate_ids(request, engine, prompt_ids, sampling)
        return _sse_response(_stream_from_result(result, "chat", request_id, created, engine.tokenizer))

    result = await _generate_ids(request, engine, prompt_ids, sampling)
    return ChatCompletionResponse(
        id=request_id,
        created=created,
        model=SERVED_MODEL_ID,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=_usage(result),
    )
