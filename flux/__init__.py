"""Flux — a lightweight LLM inference engine.

Flux serves causal language models over an OpenAI-compatible HTTP API. The core
:mod:`flux.engine` implements a batched token-by-token decode loop with a
per-batch KV cache, cooperative scheduling, and streaming output.
"""

__version__ = "0.1.0"

from flux.config import EngineConfig, ServerConfig

__all__ = ["EngineConfig", "ServerConfig", "__version__"]
