import torch

from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.naive_engine import NaiveEngine
from flux.engine.types import SamplingParams


def _engine() -> NaiveEngine:
    return NaiveEngine(FakeLM(), FakeTokenizer(), device="cpu")


def test_greedy_is_deterministic() -> None:
    engine = _engine()
    params = SamplingParams(max_tokens=4, temperature=0.0)
    first = engine.generate("hello", params)
    second = engine.generate("hello", params)
    assert first.output_token_ids == second.output_token_ids
    assert first.finish_reason == "length"
    assert first.completion_tokens == 4
    assert first.text


def test_naive_forward_grows_full_sequence() -> None:
    model = FakeLM()
    engine = NaiveEngine(model, FakeTokenizer(), device="cpu")
    engine.generate("hi", SamplingParams(max_tokens=4, temperature=0.0))
    shapes = [call["input_ids_shape"] for call in model.calls]
    assert len(shapes) == 4
    lengths = [shape[1] for shape in shapes]
    assert lengths == list(range(lengths[0], lengths[0] + 4))
    for call in model.calls:
        assert call["kwargs"].get("use_cache") is False
        assert call["kwargs"].get("past_key_values") is None


def test_empty_prompt_rejected() -> None:
    engine = _engine()
    try:
        engine.generate("", SamplingParams(max_tokens=1))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_stop_on_eos() -> None:
    class EosOnce(FakeLM):
        def forward(self, input_ids: torch.Tensor, **kwargs):
            self.calls.append({"input_ids_shape": tuple(input_ids.shape), "kwargs": dict(kwargs)})
            batch, seq = input_ids.shape
            logits = torch.zeros(batch, seq, self.vocab_size)
            logits[:, -1, self.eos_token_id] = 10.0
            from flux.engine.fake_lm import FakeLMOutput

            return FakeLMOutput(logits=logits)

    engine = NaiveEngine(EosOnce(), FakeTokenizer(), device="cpu")
    result = engine.generate("hi", SamplingParams(max_tokens=8, temperature=0.0))
    assert result.finish_reason == "stop"
    assert result.output_token_ids == [0]
