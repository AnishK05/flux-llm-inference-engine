"""Greedy / temperature / nucleus (top-p) sampling over next-token logits."""

from __future__ import annotations

import torch

from flux.engine.types import SamplingParams


def sample_logits(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample token ids from `[..., vocab]` logits. Returns shape `[batch]`."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError(f"expected logits rank 1 or 2, got {tuple(logits.shape)}")

    if params.temperature <= 0:
        return logits.argmax(dim=-1)

    scaled = logits / max(float(params.temperature), 1e-8)
    if 0.0 < params.top_p < 1.0:
        rows = [_nucleus_mask_row(row, params.top_p) for row in scaled]
        scaled = torch.stack(rows, dim=0)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _nucleus_mask_row(logits_row: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits_row, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    # Keep the token that crosses the threshold (shift the mask right).
    shifted = remove.clone()
    shifted[..., 1:] = remove[..., :-1]
    shifted[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(shifted, float("-inf"))
    restored = torch.full_like(logits_row, float("-inf"))
    restored.scatter_(0, sorted_idx, sorted_logits)
    return restored
