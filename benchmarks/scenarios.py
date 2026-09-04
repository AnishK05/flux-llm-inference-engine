"""CPU-sized loadgen scenarios from the implementation plan."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt_tokens: int | tuple[int, int]
    max_tokens: int | tuple[int, int]
    concurrencies: tuple[int, ...]
    reason: str
    mixed: bool = False


SCENARIOS: dict[str, Scenario] = {
    "short_chat": Scenario(
        name="short_chat",
        prompt_tokens=64,
        max_tokens=32,
        concurrencies=(1, 4, 8),
        reason="interactive",
    ),
    "long_prompt": Scenario(
        name="long_prompt",
        prompt_tokens=512,
        max_tokens=16,
        concurrencies=(1, 4),
        reason="prefill / KV-cache win",
    ),
    "mixed": Scenario(
        name="mixed",
        prompt_tokens=(64, 512),
        max_tokens=(16, 64),
        concurrencies=(8,),
        reason="continuous batching",
        mixed=True,
    ),
    "naive_vs_flux": Scenario(
        name="naive_vs_flux",
        prompt_tokens=64,
        max_tokens=32,
        concurrencies=(4,),
        reason="resume graph (do not use concurrency 32)",
    ),
    "soak_200": Scenario(
        name="soak_200",
        prompt_tokens=32,
        max_tokens=8,
        concurrencies=(200,),
        reason="queue + 429 + live UI; not a latency SLO",
    ),
}


def prompt_of_tokens(n: int) -> str:
    return ("alpha " * max(1, n)).strip()
