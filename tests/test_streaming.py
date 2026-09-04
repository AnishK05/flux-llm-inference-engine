import asyncio
import json
import time

import httpx
from fastapi.testclient import TestClient

from flux.config import SERVED_MODEL_ID, Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.scheduler import RequestQueue, Scheduler
from flux.engine.sequence import Sequence
from flux.engine.tokenizer import decode_delta, encode_text
from flux.engine.types import SamplingParams
from flux.engine.worker import ContinuousWorker
from flux.server.app import create_app
from flux.server.sse import chat_chunk, completion_chunk, sse_data


class SlowFakeLM(FakeLM):
    def __init__(self, delay_s: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.delay_s = delay_s

    def forward(self, *args, **kwargs):
        time.sleep(self.delay_s)
        return super().forward(*args, **kwargs)


def _parse_sse(body: str) -> list[dict | str]:
    events: list[dict | str] = []
    for block in body.split("\n\n"):
        line = block.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


def test_sse_pack_helpers() -> None:
    raw = sse_data(completion_chunk("cmpl-1", 1, "Hi")).decode()
    assert raw.startswith("data: {")
    assert raw.endswith("\n\n")
    assert sse_data("[DONE]") == b"data: [DONE]\n\n"
    chat = chat_chunk("chatcmpl-1", 1, {"content": "x"})
    assert chat["object"] == "chat.completion.chunk"


def test_decode_delta_is_prefix_stable() -> None:
    tok = FakeTokenizer()
    ids = tok.encode("ab")
    full, first = decode_delta(tok, ids[:1], "")
    later, delta = decode_delta(tok, ids, full)
    assert first
    assert later.startswith(full)
    assert delta == later[len(full) :]


def test_stream_completions_http(client_continuous: TestClient) -> None:
    with client_continuous.stream(
        "POST",
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "hello", "max_tokens": 4, "temperature": 0, "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(response.read().decode())
    assert events[-1] == "[DONE]"
    texts = [e["choices"][0]["text"] for e in events[:-1] if isinstance(e, dict)]
    assert any(texts)
    assert events[-2]["choices"][0]["finish_reason"] in {"length", "stop"}


def test_stream_chat_http(client_continuous: TestClient) -> None:
    with client_continuous.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": SERVED_MODEL_ID,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 3,
            "temperature": 0,
            "stream": True,
        },
    ) as response:
        events = _parse_sse(response.read().decode())
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert events[-1] == "[DONE]"
    contents = [
        e["choices"][0]["delta"].get("content")
        for e in events
        if isinstance(e, dict) and "content" in e["choices"][0].get("delta", {})
    ]
    assert any(contents)


def test_first_token_arrives_before_finish() -> None:
    async def main() -> None:
        engine = CachedEngine(SlowFakeLM(0.04), FakeTokenizer())
        worker = ContinuousWorker(engine, Scheduler(RequestQueue(8), max_batch=8))
        seq = Sequence(
            prompt_ids=encode_text(engine.tokenizer, "hello"),
            sampling=SamplingParams(max_tokens=6, temperature=0.0),
        )
        task = asyncio.create_task(worker.run())
        try:
            worker.scheduler.submit(seq)
            token = await asyncio.wait_for(seq.token_queue.get(), timeout=2)
            assert token is not None
            assert not seq.finished_event.is_set()
            await seq.finished_event.wait()
        finally:
            worker.request_stop()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())


def test_abort_frees_kv_blocks_worker() -> None:
    async def main() -> None:
        from flux.engine.block_pool import BlockPool
        from flux.engine.sequence import SequenceStatus

        engine = CachedEngine(SlowFakeLM(0.04), FakeTokenizer())
        pool = BlockPool(4, 16)
        worker = ContinuousWorker(engine, Scheduler(RequestQueue(8), max_batch=8, pool=pool))
        seq = Sequence(
            prompt_ids=encode_text(engine.tokenizer, "hello"),
            sampling=SamplingParams(max_tokens=12, temperature=0.0),
        )
        task = asyncio.create_task(worker.run())
        try:
            worker.scheduler.submit(seq)
            token = await asyncio.wait_for(seq.token_queue.get(), timeout=2)
            assert token is not None
            assert pool.used_blocks >= 1
            seq.request_abort()
            await asyncio.wait_for(seq.finished_event.wait(), timeout=2)
            assert seq.status == SequenceStatus.ABORTED
            assert pool.used_blocks == 0
        finally:
            worker.request_stop()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())


def test_abort_frees_kv_blocks() -> None:
    async def main() -> None:
        engine = CachedEngine(SlowFakeLM(0.05), FakeTokenizer())
        settings = Settings(
            load_model=False,
            serve_engine="continuous",
            num_kv_blocks="4",
            block_size=16,
        )
        app = create_app(settings, engine=engine)
        async with app.router.lifespan_context(app):
            pool = app.state.block_pool
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream(
                    "POST",
                    "/v1/completions",
                    json={
                        "model": SERVED_MODEL_ID,
                        "prompt": "hello stream abort",
                        "max_tokens": 12,
                        "temperature": 0,
                        "stream": True,
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data:") and "[DONE]" not in line and "{" in line:
                            break
                await asyncio.sleep(0.25)
            assert pool.used_blocks == 0
            assert app.state.worker.stats.running == []

    asyncio.run(main())
