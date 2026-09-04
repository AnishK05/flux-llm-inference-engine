import torch

from flux.engine.sampler import sample_logits
from flux.engine.types import SamplingParams


def test_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    greedy = sample_logits(logits, SamplingParams(temperature=0.0))
    still_greedy = sample_logits(logits, SamplingParams(temperature=0.0, top_p=0.1))
    assert int(greedy.item()) == 1
    assert int(still_greedy.item()) == 1


def test_top_p_keeps_nucleus() -> None:
    # Token 2 dominates; token 1 is a distant second; 0 is tiny.
    logits = torch.tensor([[0.0, 2.0, 8.0]])
    torch.manual_seed(0)
    token = sample_logits(logits, SamplingParams(temperature=1.0, top_p=0.5))
    assert int(token.item()) == 2


def test_temperature_positive_is_stochastic_but_in_vocab() -> None:
    logits = torch.tensor([[1.0, 1.1, 0.9, 1.05]])
    torch.manual_seed(1)
    token = sample_logits(logits, SamplingParams(temperature=0.8, top_p=1.0))
    assert 0 <= int(token.item()) < 4


def test_rank1_logits() -> None:
    logits = torch.tensor([0.0, 5.0, 1.0])
    token = sample_logits(logits, SamplingParams(temperature=0.0))
    assert int(token.item()) == 1
