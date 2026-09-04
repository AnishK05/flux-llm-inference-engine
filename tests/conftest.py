"""Shared pytest fixtures.

A single engine is loaded once per test session to avoid repeatedly paying the
model-load cost. Tests default to the small ``distilgpt2`` model, which the
environment setup already caches; override with ``FLUX_TEST_MODEL``.
"""

from __future__ import annotations

import os

import pytest

from flux.config import EngineConfig
from flux.engine import InferenceEngine

TEST_MODEL = os.environ.get("FLUX_TEST_MODEL", "distilgpt2")


@pytest.fixture(scope="session")
def engine() -> InferenceEngine:
    eng = InferenceEngine(EngineConfig(model=TEST_MODEL, max_batch_size=4))
    eng.load()
    assert eng.wait_until_ready(timeout=120)
    yield eng
    eng.shutdown()
