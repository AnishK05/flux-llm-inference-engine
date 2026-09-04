"""Configuration objects for the Flux engine and HTTP server.

All values can be overridden through environment variables so the same code runs
identically in local development and in a Cloud Agent environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class EngineConfig:
    """Runtime configuration for :class:`flux.engine.InferenceEngine`."""

    model: str = "distilgpt2"
    device: str = "cpu"
    dtype: str = "float32"
    # Maximum number of requests decoded together in a single static batch.
    max_batch_size: int = 4
    # Hard ceiling on generated tokens per request (independent of the request's
    # own ``max_tokens``) to protect the server from runaway generations.
    max_new_tokens_cap: int = 512
    # Upper bound on prompt length in tokens; longer prompts are rejected.
    max_prompt_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "EngineConfig":
        return cls(
            model=_env_str("FLUX_MODEL", cls.model),
            device=_env_str("FLUX_DEVICE", cls.device),
            dtype=_env_str("FLUX_DTYPE", cls.dtype),
            max_batch_size=_env_int("FLUX_MAX_BATCH_SIZE", cls.max_batch_size),
            max_new_tokens_cap=_env_int("FLUX_MAX_NEW_TOKENS", cls.max_new_tokens_cap),
            max_prompt_tokens=_env_int("FLUX_MAX_PROMPT_TOKENS", cls.max_prompt_tokens),
        )


@dataclass
class ServerConfig:
    """Configuration for the FastAPI/uvicorn server process."""

    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "ServerConfig":
        return cls(
            host=_env_str("FLUX_HOST", cls.host),
            port=_env_int("FLUX_PORT", cls.port),
        )
