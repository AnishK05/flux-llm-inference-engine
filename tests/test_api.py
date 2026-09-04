"""HTTP API tests using FastAPI's TestClient against a shared engine."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from flux.server import build_chat_prompt, create_app
from flux.protocol import ChatMessage


@pytest.fixture(scope="module")
def client(engine):
    app = create_app(engine)
    with TestClient(app) as c:
        yield c


def test_health(client, engine):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == engine.config.model


def test_list_models(client, engine):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert any(card["id"] == engine.config.model for card in data)


def test_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Flux" in resp.text


def test_completion_non_stream(client):
    resp = client.post(
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 6, "temperature": 0.0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["finish_reason"] in {"length", "stop"}
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["total_tokens"] >= body["usage"]["prompt_tokens"]


def test_completion_stream(client):
    with client.stream(
        "POST",
        "/v1/completions",
        json={"prompt": "Hello world", "max_tokens": 6, "temperature": 0.0, "stream": True},
    ) as resp:
        assert resp.status_code == 200
        chunks, saw_done = [], False
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                saw_done = True
                continue
            chunks.append(json.loads(payload))
    assert saw_done
    assert any(c["choices"][0]["text"] for c in chunks)


def test_chat_completion(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 12,
            "temperature": 0.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["role"] == "assistant"


def test_prompt_too_long_returns_413(client, engine):
    long_prompt = "word " * (engine.config.max_prompt_tokens + 50)
    resp = client.post("/v1/completions", json={"prompt": long_prompt, "max_tokens": 4})
    assert resp.status_code == 413


def test_build_chat_prompt_format():
    prompt = build_chat_prompt(
        [
            ChatMessage(role="system", content="Be nice"),
            ChatMessage(role="user", content="Hi"),
        ]
    )
    assert prompt.endswith("Assistant:")
    assert "System: Be nice" in prompt
    assert "User: Hi" in prompt
