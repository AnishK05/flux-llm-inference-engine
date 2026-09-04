from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from flux.config import Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.server.app import create_app


def _client(engine, **settings_kw) -> Iterator[TestClient]:
    app = create_app(settings=Settings(load_model=False, **settings_kw), engine=engine)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_no_model() -> Iterator[TestClient]:
    yield from _client(None, serve_engine="cached")


@pytest.fixture
def client_fake() -> Iterator[TestClient]:
    yield from _client(CachedEngine(FakeLM(), FakeTokenizer()), serve_engine="cached")


@pytest.fixture
def client_continuous() -> Iterator[TestClient]:
    yield from _client(CachedEngine(FakeLM(), FakeTokenizer()), serve_engine="continuous")


@pytest.fixture
def client_queued() -> Iterator[TestClient]:
    yield from _client(CachedEngine(FakeLM(), FakeTokenizer()), serve_engine="queued")
