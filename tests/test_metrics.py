from fastapi.testclient import TestClient

from flux.config import SERVED_MODEL_ID, Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.server.app import create_app


def test_metrics_exposes_flux_series(client_continuous: TestClient) -> None:
    response = client_continuous.get("/metrics")
    assert response.status_code == 200
    body = response.text
    assert "flux_waiting" in body
    assert "flux_running" in body
    assert "flux_kv_blocks_used" in body
    assert "flux_rss_bytes" in body
    assert "flux_decode_batch_size" in body


def test_metrics_move_after_generate(client_continuous: TestClient) -> None:
    before = client_continuous.get("/metrics").text
    response = client_continuous.post(
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "hello", "max_tokens": 3, "temperature": 0},
    )
    assert response.status_code == 200
    assert "x-flux-ttft-ms" in response.headers
    after = client_continuous.get("/metrics").text
    assert "flux_output_tokens_total" in after
    assert "flux_ttft_seconds" in after
    assert "flux_http_requests_total" in after
    assert after != before or "flux_output_tokens_total" in before


def test_metrics_without_worker() -> None:
    app = create_app(
        settings=Settings(load_model=False, serve_engine="cached"),
        engine=CachedEngine(FakeLM(), FakeTokenizer()),
    )
    with TestClient(app) as client:
        body = client.get("/metrics").text
    assert "flux_waiting 0" in body
