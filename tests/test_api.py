from fastapi.testclient import TestClient

from flux.config import HF_MODEL_ID, SERVED_MODEL_ID, Settings


def test_health_without_model(client_no_model: TestClient) -> None:
    response = client_no_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "cpu"
    assert body["model_loaded"] is False
    assert body["model_id"]
    assert "intra_threads" in body
    assert body["rss_bytes"] > 0


def test_ready_503_until_loaded(client_no_model: TestClient) -> None:
    response = client_no_model.get("/ready")
    assert response.status_code == 503


def test_completions_503_without_model(client_no_model: TestClient) -> None:
    response = client_no_model.post(
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "hello", "max_tokens": 4},
    )
    assert response.status_code == 503


def test_completions_400_empty_prompt(client_fake: TestClient) -> None:
    response = client_fake.post(
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "  ", "max_tokens": 4},
    )
    assert response.status_code == 400


def test_completions_404_unknown_model(client_fake: TestClient) -> None:
    response = client_fake.post(
        "/v1/completions",
        json={"model": "gpt-x", "prompt": "hello", "max_tokens": 4},
    )
    assert response.status_code == 404


def test_completions_happy_path(client_fake: TestClient) -> None:
    response = client_fake.post(
        "/v1/completions",
        json={"model": HF_MODEL_ID, "prompt": "hello", "max_tokens": 4, "temperature": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == SERVED_MODEL_ID
    assert body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] in {"length", "stop"}
    assert body["usage"]["completion_tokens"] >= 1
    assert body["usage"]["prompt_tokens"] >= 1


def test_max_tokens_capped(client_fake: TestClient) -> None:
    response = client_fake.post(
        "/v1/completions",
        json={"model": SERVED_MODEL_ID, "prompt": "hello", "max_tokens": 1000, "temperature": 0},
    )
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] <= Settings().max_new_tokens_cap


def test_ready_ok_with_engine(client_fake: TestClient) -> None:
    response = client_fake.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_health_reports_cached_engine(client_fake: TestClient) -> None:
    body = client_fake.get("/health").json()
    assert body["engine_name"] == "cached"
    assert body["serve_engine"] == "cached"


def test_chat_completions_happy_path(client_fake: TestClient) -> None:
    response = client_fake.post(
        "/v1/chat/completions",
        json={
            "model": SERVED_MODEL_ID,
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 4,
            "temperature": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["prompt_tokens"] >= 1


def test_chat_completions_400_empty_messages(client_fake: TestClient) -> None:
    response = client_fake.post(
        "/v1/chat/completions",
        json={"model": SERVED_MODEL_ID, "messages": [], "max_tokens": 4},
    )
    assert response.status_code == 400
