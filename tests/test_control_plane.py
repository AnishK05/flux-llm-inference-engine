from fastapi.testclient import TestClient

from flux.config import SERVED_MODEL_ID, Settings
from flux.control import RecentTtft, TokenRate
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.redis_client import ControlPlane
from flux.server.app import create_app


def test_recent_ttft_p50() -> None:
    ring = RecentTtft(window_s=60)
    assert ring.p50_ms() is None
    ring.add(0.10)
    ring.add(0.20)
    ring.add(0.30)
    assert 150 <= (ring.p50_ms() or 0) <= 250


def test_token_rate_window() -> None:
    rate = TokenRate(window_s=5)
    assert rate.observe(0) == 0.0
    first = rate.observe(10)
    assert first >= 0.0


def test_job_store_memory() -> None:
    plane = ControlPlane()
    plane.put_job("abc", {"status": "running"})
    job = plane.get_job("abc")
    assert job is not None
    assert job["status"] == "running"
    assert plane.get_job("missing") is None


def test_rate_limit_with_fake_redis() -> None:
    plane = ControlPlane()
    plane.enabled = True

    class FakeRedis:
        def __init__(self) -> None:
            self.n = 0

        def incr(self, _key: str) -> int:
            self.n += 1
            return self.n

        def expire(self, _key: str, _ttl: int) -> None:
            return None

    plane._redis = FakeRedis()
    assert plane.allow("1.1.1.1", limit=2)
    assert plane.allow("1.1.1.1", limit=2)
    assert plane.allow("1.1.1.1", limit=2) is False


def test_cors_preflight() -> None:
    app = create_app(
        settings=Settings(load_model=False, serve_engine="continuous"),
        engine=CachedEngine(FakeLM(), FakeTokenizer()),
    )
    with TestClient(app) as client:
        response = client.options(
            "/admin/stats",
            headers={
                "origin": "http://localhost:3000",
                "access-control-request-method": "GET",
            },
        )
    assert response.status_code in {200, 204}
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_admin_stats_live_fields(client_continuous: TestClient) -> None:
    body = client_continuous.get("/admin/stats").json()
    assert "waiting_ids" in body
    assert "running_ids" in body
    assert "rss_bytes" in body
    assert "tok_s" in body
    assert "ttft_p50_ms" in body
    assert body["waiting_ids"] == []
    assert body["running_ids"] == []


def test_admin_bench_serves_last_run(client_continuous: TestClient) -> None:
    response = client_continuous.get("/admin/bench")
    assert response.status_code == 200
    body = response.json()
    assert "runs" in body
    assert body["runs"]


def test_job_and_abort(client_continuous: TestClient) -> None:
    response = client_continuous.post(
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "hello", "max_tokens": 2, "temperature": 0},
    )
    assert response.status_code == 200
    request_id = response.json()["id"]
    job = client_continuous.get(f"/v1/requests/{request_id}")
    assert job.status_code == 200
    assert job.json()["status"] == "finished"
    missing = client_continuous.post("/admin/abort/does-not-exist")
    assert missing.status_code == 404
