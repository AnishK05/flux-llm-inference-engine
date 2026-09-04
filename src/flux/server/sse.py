"""OpenAI-shaped Server-Sent Event chunks."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from flux.config import SERVED_MODEL_ID
from flux.engine.sequence import Sequence
from flux.engine.tokenizer import decode_delta

DONE = b"data: [DONE]\n\n"


def sse_data(payload: dict[str, Any] | str) -> bytes:
    if payload == "[DONE]":
        return DONE
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def completion_chunk(
    request_id: str,
    created: int,
    text: str,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": SERVED_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }


def chat_chunk(
    request_id: str,
    created: int,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": SERVED_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


async def iter_text_deltas(seq: Sequence, tokenizer: Any) -> AsyncIterator[str]:
    """Yield decoded text as soon as each token is pushed onto `seq.token_queue`."""
    prev = ""
    while True:
        item = await _next_token(seq)
        if item is None:
            break
        prev, delta = decode_delta(tokenizer, seq.output_ids, prev)
        if delta:
            yield delta
    _, delta = decode_delta(tokenizer, seq.output_ids, prev)
    if delta:
        yield delta


async def _next_token(seq: Sequence) -> int | None:
    while True:
        if seq.finished_event.is_set() and seq.token_queue.empty():
            return None
        try:
            item = await asyncio.wait_for(seq.token_queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        return item
