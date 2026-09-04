"""Token sampling utilities.

These functions operate on a single row of logits at a time so that every
request in a batch can carry its own sampling parameters (temperature, top-k,
top-p, repetition penalty). Batches in Flux are small, so a per-row Python loop
in the engine is both clear and fast enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class SamplingParams:
    """Per-request sampling configuration."""

    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    def normalized(self) -> "SamplingParams":
        """Clamp values into safe ranges without mutating the original."""

        return SamplingParams(
            temperature=max(float(self.temperature), 0.0),
            top_k=max(int(self.top_k), 0),
            top_p=min(max(float(self.top_p), 0.0), 1.0),
            repetition_penalty=max(float(self.repetition_penalty), 0.0) or 1.0,
            seed=self.seed,
        )


def apply_repetition_penalty(
    logits: torch.Tensor, generated: Sequence[int], penalty: float
) -> torch.Tensor:
    """Penalize tokens that already appeared, following the CTRL formulation."""

    if penalty == 1.0 or not generated:
        return logits
    unique = torch.tensor(sorted(set(generated)), dtype=torch.long, device=logits.device)
    selected = logits.index_select(0, unique)
    # Positive logits are divided by the penalty, negative logits multiplied.
    adjusted = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits = logits.clone()
    logits.index_copy_(0, unique, adjusted)
    return logits


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    kth_value = torch.topk(logits, top_k).values[-1]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        # Degenerate nucleus: keep only the single most likely token.
        top_index = int(torch.argmax(logits))
        filtered = torch.full_like(logits, float("-inf"))
        filtered[top_index] = logits[top_index]
        return filtered

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    # Keep tokens up to and including the one that crosses the top_p threshold.
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    restored = torch.full_like(logits, float("-inf"))
    restored.scatter_(0, sorted_indices, sorted_logits)
    return restored


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generated: Sequence[int],
    generator: torch.Generator | None = None,
) -> int:
    """Return the next token id sampled from a single row of ``logits``."""

    params = params.normalized()
    logits = apply_repetition_penalty(logits, generated, params.repetition_penalty)

    # Temperature of 0 (or extremely small) means greedy/argmax decoding.
    if params.temperature <= 1e-6:
        return int(torch.argmax(logits))

    logits = logits / params.temperature
    logits = top_k_filter(logits, params.top_k)
    logits = top_p_filter(logits, params.top_p)

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_token)
