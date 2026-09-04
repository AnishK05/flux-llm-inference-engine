from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from flux.config import Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.server.app import create_app


def _client(engine) -> Iterator[TestClient]:
    app = create_app(settings=Settings(load_model=False), engine=engine)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def client_no_model() -> Iterator[TestClient]:
    yield from _client(None)


@pytest.fixture
def client_fake() -> Iterator[TestClient]:
    yield from _client(CachedEngine(FakeLM(), FakeTokenizer()))
