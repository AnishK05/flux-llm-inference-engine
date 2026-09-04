"""FastAPI application exposing an OpenAI-compatible API over the Flux engine."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from flux.config import EngineConfig, ServerConfig
from flux.engine import InferenceEngine, PromptTooLongError, RequestHandle
from flux.protocol import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelList,
    UsageInfo,
)
from flux.sampling import SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("flux.server")

_WEB_DIR = Path(__file__).parent / "web"
# Default stop sequences that keep chat models from continuing the dialogue on
# behalf of the user when the base model has no dedicated chat template.
_CHAT_STOPS = ["\nUser:", "\nSystem:", "\nuser:", "\nsystem:"]


def _normalize_stop(stop) -> List[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def build_chat_prompt(messages: List[ChatMessage]) -> str:
    """Render chat messages into a plain-text prompt for a base LM."""

    lines: List[str] = []
    for message in messages:
        role = message.role.capitalize()
        lines.append(f"{role}: {message.content.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)


def create_app(engine: Optional[InferenceEngine] = None) -> FastAPI:
    # Track ownership: only an engine we created here should be shut down when
    # the app stops. An injected engine (e.g. a shared test fixture) is managed
    # by its owner.
    owns_engine = engine is None
    engine = engine or InferenceEngine(EngineConfig.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not engine.is_ready:
            await asyncio.to_thread(engine.load)
        app.state.engine = engine
        yield
        if owns_engine:
            engine.shutdown()

    app = FastAPI(title="Flux LLM Inference Engine", version="0.1.0", lifespan=lifespan)

    async def _drain(handle: RequestHandle) -> AsyncGenerator[dict, None]:
        """Yield engine messages for a request without blocking the event loop."""

        while True:
            message = await asyncio.to_thread(handle.out_queue.get)
            yield message
            if message.get("type") == "done":
                break

    def _sampling_from(req) -> SamplingParams:
        return SamplingParams(
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            seed=req.seed,
        )

    # ---------------------------------------------------------------- routes
    @app.get("/health")
    async def health():
        return {"status": "ok" if engine.is_ready else "loading", **engine.info()}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=engine.config.model)])

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = _WEB_DIR / "index.html"
        if html.exists():
            return HTMLResponse(html.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Flux</h1><p>UI asset missing.</p>")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        if isinstance(req.prompt, list):
            if len(req.prompt) != 1:
                raise HTTPException(400, "Only a single prompt is supported")
            prompt = req.prompt[0]
        else:
            prompt = req.prompt

        stop = _normalize_stop(req.stop)
        try:
            handle = engine.submit(prompt, _sampling_from(req), req.max_tokens, stop)
        except PromptTooLongError as exc:
            raise HTTPException(413, str(exc))

        model_name = engine.config.model
        if req.stream:
            async def event_stream() -> AsyncGenerator[str, None]:
                async for message in _drain(handle):
                    if message["type"] == "token":
                        chunk = CompletionResponse(
                            model=model_name,
                            choices=[CompletionChoice(text=message["text"])],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    elif message["type"] == "done":
                        chunk = CompletionResponse(
                            model=model_name,
                            choices=[CompletionChoice(finish_reason=message["finish_reason"])],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                        yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        text, finish_reason, completion_tokens = "", "stop", 0
        async for message in _drain(handle):
            if message["type"] == "token":
                text += message["text"]
            elif message["type"] == "done":
                finish_reason = message["finish_reason"]
                completion_tokens = message["completion_tokens"]

        return CompletionResponse(
            model=model_name,
            choices=[CompletionChoice(text=text, finish_reason=finish_reason)],
            usage=UsageInfo(
                prompt_tokens=handle.prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=handle.prompt_tokens + completion_tokens,
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        prompt = build_chat_prompt(req.messages)
        stop = _normalize_stop(req.stop) + _CHAT_STOPS
        try:
            handle = engine.submit(prompt, _sampling_from(req), req.max_tokens, stop)
        except PromptTooLongError as exc:
            raise HTTPException(413, str(exc))

        model_name = engine.config.model
        if req.stream:
            async def event_stream() -> AsyncGenerator[str, None]:
                async for message in _drain(handle):
                    if message["type"] == "token":
                        chunk = {
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {"content": message["text"]}, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif message["type"] == "done":
                        chunk = {
                            "object": "chat.completion.chunk",
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": message["finish_reason"]}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        text, finish_reason, completion_tokens = "", "stop", 0
        async for message in _drain(handle):
            if message["type"] == "token":
                text += message["text"]
            elif message["type"] == "done":
                finish_reason = message["finish_reason"]
                completion_tokens = message["completion_tokens"]

        return ChatCompletionResponse(
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=text.strip()),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=handle.prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=handle.prompt_tokens + completion_tokens,
            ),
        )

    return app


app = create_app()
