from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 16
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False


@dataclass
class GenerateResult:
    text: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    latency_s: float
    tokens_per_second: float
    ttft_s: float = 0.0
    tpot_s: float | None = None
    engine: str = ""
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
