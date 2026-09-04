import asyncio

import httpx
from fastapi.testclient import TestClient

from flux.config import SERVED_MODEL_ID, Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.scheduler import QueueFull, RequestQueue, Scheduler
from flux.engine.sequence import Sequence
from flux.engine.types import SamplingParams
from flux.server.app import create_app


def test_health_reports_continuous_engine(client_continuous: TestClient) -> None:
    body = client_continuous.get("/health").json()
    assert body["engine_name"] == "continuous"
    assert body["serve_engine"] == "continuous"


def test_health_reports_queued_engine(client_queued: TestClient) -> None:
    body = client_queued.get("/health").json()
    assert body["engine_name"] == "queued"
    assert body["serve_engine"] == "queued"


def test_continuous_completions_happy_path(client_continuous: TestClient) -> None:
    response = client_continuous.post(
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "hello", "max_tokens": 4, "temperature": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"]
    assert body["usage"]["completion_tokens"] >= 1


def test_queued_chat_completions_happy_path(client_queued: TestClient) -> None:
    response = client_queued.post(
        "/v1/chat/completions",
        json={
            "model": SERVED_MODEL_ID,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4,
            "temperature": 0,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]


def test_admin_stats_shape(client_continuous: TestClient) -> None:
    body = client_continuous.get("/admin/stats").json()
    assert body["waiting"] == 0
    assert body["running"] == 0
    assert body["in_flight"] == 0
    assert body["max_waiting"] == 256
    assert body["max_batch_size"] == 8
    assert body["serve_engine"] == "continuous"
    assert "tokens_generated" in body
    assert "kv_blocks_used" in body
    assert "kv_blocks_free" in body
    assert "kv_blocks_total" in body
    assert body["kv_blocks_used"] == 0
    assert body["kv_blocks_total"] >= 1


def test_admin_stats_after_completion(client_continuous: TestClient) -> None:
    response = client_continuous.post(
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "hello", "max_tokens": 3, "temperature": 0},
    )
    assert response.status_code == 200
    body = client_continuous.get("/admin/stats").json()
    assert body["tokens_generated"] >= 1
    assert body["waiting"] == 0
    assert body["running"] == 0
    assert body["kv_blocks_used"] == 0


def test_http_429_when_waiting_queue_full() -> None:
    engine = CachedEngine(FakeLM(), FakeTokenizer())
    settings = Settings(load_model=False, serve_engine="queued", max_waiting=0)
    app = create_app(settings, engine=engine)
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": SERVED_MODEL_ID, "prompt": "hello", "max_tokens": 2, "temperature": 0},
        )
    assert response.status_code == 429
    assert response.headers.get("retry-after") == "1"
    assert "full" in response.json()["detail"]


def test_prompt_exceeds_max_seq_len_400(client_fake: TestClient) -> None:
    app = client_fake.app
    app.state.settings = Settings(load_model=False, serve_engine="cached", max_seq_len=3)
    response = client_fake.post(
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "hello world", "max_tokens": 2, "temperature": 0},
    )
    assert response.status_code == 400
    assert "max_seq_len" in response.json()["detail"]


def test_fifty_concurrent_requests_complete() -> None:
    async def main() -> None:
        engine = CachedEngine(FakeLM(), FakeTokenizer())
        settings = Settings(load_model=False, serve_engine="continuous", max_waiting=256)
        app = create_app(settings, engine=engine)
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                async def one(i: int):
                    return await client.post(
                        "/v1/completions",
                        json={
                            "model": SERVED_MODEL_ID,
                            "prompt": f"hello {i}",
                            "max_tokens": 2,
                            "temperature": 0,
                        },
                    )

                responses = await asyncio.gather(*[one(i) for i in range(50)])
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["choices"][0]["text"] for r in responses)

    asyncio.run(main())


def test_queue_full_unit_matches_http() -> None:
    scheduler = Scheduler(RequestQueue(max_waiting=0), max_batch=8)
    seq = Sequence(
        prompt_ids=FakeTokenizer()("a", return_tensors="pt")["input_ids"],
        sampling=SamplingParams(),
    )
    try:
        scheduler.submit(seq)
        raised = False
    except QueueFull:
        raised = True
    assert raised
