import os

import pytest

from flux.config import HF_MODEL_ID, Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.kv_utils import describe_cache
from flux.engine.model_loader import load_causal_lm
from flux.engine.naive_engine import NaiveEngine
from flux.engine.tokenizer import encode_chat, stop_token_ids
from flux.engine.types import SamplingParams
from flux.runtime import apply_thread_caps

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("FLUX_RUN_INTEGRATION") != "1",
        reason="set FLUX_RUN_INTEGRATION=1 to download Qwen and run this",
    ),
]


@pytest.fixture(scope="module")
def qwen_pair():
    apply_thread_caps("auto")
    settings = Settings(load_model=True, model=HF_MODEL_ID, device="cpu", dtype="fp32")
    model, tokenizer = load_causal_lm(settings)
    naive = NaiveEngine(model, tokenizer, device="cpu")
    cached = CachedEngine(model, tokenizer, device="cpu")
    return naive, cached, model, tokenizer


def test_qwen_naive_max_tokens_4(qwen_pair) -> None:
    naive, _, _, _ = qwen_pair
    result = naive.generate("Say hi.", SamplingParams(max_tokens=4, temperature=0.0))
    assert result.completion_tokens == 4 or result.finish_reason == "stop"
    assert result.prompt_tokens >= 1
    assert result.latency_s > 0
    assert result.text is not None


def test_qwen_cached_matches_naive_greedy(qwen_pair) -> None:
    naive, cached, _, _ = qwen_pair
    params = SamplingParams(max_tokens=8, temperature=0.0)
    left = naive.generate("The capital of France is", params)
    right = cached.generate("The capital of France is", params)
    assert left.output_token_ids == right.output_token_ids
    assert left.finish_reason == right.finish_reason


def test_qwen_kv_shapes_match_config(qwen_pair) -> None:
    _, cached, model, tokenizer = qwen_pair
    prompt_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
    _, cache = cached.prefill(prompt_ids)
    info = describe_cache(cache)
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    assert info["n_layers"] == config.num_hidden_layers
    assert info["n_kv_heads"] == config.num_key_value_heads
    assert info["head_dim"] == head_dim
    assert info["seq_len"] == prompt_ids.shape[1]


def test_qwen_chat_template_applied(qwen_pair) -> None:
    _, cached, _, tokenizer = qwen_pair
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Name one color."},
    ]
    ids = encode_chat(tokenizer, messages)
    text = tokenizer.decode(ids[0], skip_special_tokens=False)
    assert "You are concise." in text
    assert "Name one color." in text
    assert text != "Name one color."
    stops = stop_token_ids(tokenizer)
    assert tokenizer.eos_token_id in stops
    result = cached.generate_chat(messages, SamplingParams(max_tokens=8, temperature=0.0))
    assert result.prompt_tokens == ids.shape[1]
    assert result.completion_tokens >= 1
