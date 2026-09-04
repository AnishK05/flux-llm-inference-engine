"""Unit tests for the sampling utilities (no model required)."""

from __future__ import annotations

import torch

from flux.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    sample_token,
    top_k_filter,
    top_p_filter,
)


def test_greedy_is_argmax():
    logits = torch.tensor([0.1, 5.0, 0.2, -1.0])
    params = SamplingParams(temperature=0.0)
    assert sample_token(logits, params, generated=[]) == 1


def test_top_k_filter_keeps_k_tokens():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    filtered = top_k_filter(logits.clone(), top_k=2)
    finite = torch.isfinite(filtered)
    assert int(finite.sum()) == 2
    assert finite[3] and finite[4]


def test_top_p_filter_keeps_nucleus():
    logits = torch.tensor([10.0, 1.0, 1.0, 1.0])
    filtered = top_p_filter(logits.clone(), top_p=0.5)
    # The dominant token alone exceeds the 0.5 nucleus threshold.
    assert torch.isfinite(filtered[0])
    assert int(torch.isfinite(filtered).sum()) == 1


def test_repetition_penalty_reduces_repeat_logits():
    logits = torch.tensor([2.0, 2.0, 2.0])
    penalized = apply_repetition_penalty(logits.clone(), generated=[0], penalty=2.0)
    assert penalized[0] < logits[0]
    assert penalized[1] == logits[1]


def test_seeded_sampling_is_reproducible():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    params = SamplingParams(temperature=1.0)
    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    seq1 = [sample_token(logits, params, [], g1) for _ in range(10)]
    seq2 = [sample_token(logits, params, [], g2) for _ in range(10)]
    assert seq1 == seq2
