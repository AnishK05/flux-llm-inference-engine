import torch

from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.kv_utils import cache_seq_len, describe_cache
from flux.engine.naive_engine import NaiveEngine
from flux.engine.types import SamplingParams


def _pair():
    model = FakeLM()
    tokenizer = FakeTokenizer()
    naive = NaiveEngine(model, tokenizer, device="cpu")
    cached = CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")
    return naive, cached


def test_greedy_matches_naive() -> None:
    naive, cached = _pair()
    params = SamplingParams(max_tokens=12, temperature=0.0)
    left = naive.generate("hello cache", params)
    right = cached.generate("hello cache", params)
    assert left.output_token_ids == right.output_token_ids
    assert left.finish_reason == right.finish_reason


def test_decode_uses_single_token_and_grows_cache() -> None:
    model = FakeLM(n_layers=2, n_kv_heads=2, head_dim=4)
    engine = CachedEngine(model, FakeTokenizer(), device="cpu")
    prompt = torch.tensor([[1, 4, 7]], dtype=torch.long)
    logits, cache = engine.prefill(prompt)
    assert logits.shape[-1] == model.vocab_size
    assert cache_seq_len(cache) == prompt.shape[1]
    info = describe_cache(cache)
    assert info["n_layers"] == model.n_layers
    assert info["n_kv_heads"] == model.n_kv_heads
    assert info["head_dim"] == model.head_dim

    token = logits.argmax(dim=-1)
    _, cache = engine.decode(token, cache)
    assert cache_seq_len(cache) == prompt.shape[1] + 1

    decode_calls = [c for c in model.calls if c["past_len"] > 0]
    assert decode_calls
    assert decode_calls[0]["input_ids_shape"] == (1, 1)
    assert decode_calls[0]["use_cache"] is True


def test_cached_generate_records_ttft() -> None:
    engine = CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")
    result = engine.generate("hi", SamplingParams(max_tokens=4, temperature=0.0))
    assert result.engine == "cached"
    assert result.ttft_s > 0
    assert result.tpot_s is not None
    assert result.completion_tokens == 4


def test_chat_generate_is_deterministic() -> None:
    engine = CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
    ]
    params = SamplingParams(max_tokens=6, temperature=0.0)
    first = engine.generate_chat(messages, params)
    second = engine.generate_chat(messages, params)
    third = engine.generate_chat(messages, params)
    assert first.output_token_ids == second.output_token_ids == third.output_token_ids
    assert first.prompt_tokens == second.prompt_tokens
    # Templated prompt is longer than the raw user text encoding.
    raw = engine.generate("Hello", params)
    assert first.prompt_tokens > raw.prompt_tokens
