import asyncio
import os

import pytest
import torch

from flux.config import HF_MODEL_ID, Settings
from flux.engine.cached_engine import CachedEngine
from flux.engine.kv_utils import cache_seq_len, describe_cache
from flux.engine.model_loader import load_causal_lm
from flux.engine.naive_engine import NaiveEngine
from flux.engine.scheduler import RequestQueue, Scheduler
from flux.engine.sequence import Sequence
from flux.engine.tokenizer import encode_chat, encode_text, stop_token_ids
from flux.engine.types import SamplingParams
from flux.engine.worker import ContinuousWorker
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


def test_qwen_decode_batch_mixed_lengths_match_solo(qwen_pair) -> None:
    _, cached, _, tokenizer = qwen_pair
    prompts = ["Hi", "The capital of France is"]
    ids = [tokenizer(text, return_tensors="pt")["input_ids"] for text in prompts]
    expected_tokens: list[int] = []
    expected_lens: list[int] = []
    for prompt_ids in ids:
        logits, cache = cached.prefill(prompt_ids)
        token = logits.argmax(dim=-1)
        next_logits, new_cache = cached.decode(token, cache)
        expected_tokens.append(int(next_logits.argmax(dim=-1).item()))
        expected_lens.append(cache_seq_len(new_cache))

    last_tokens = []
    caches = []
    for prompt_ids in ids:
        logits, cache = cached.prefill(prompt_ids)
        token = logits.argmax(dim=-1)
        last_tokens.append(int(token.item()))
        caches.append(cache)

    batched_logits, batched_caches = cached.decode_batch(
        torch.tensor(last_tokens, dtype=torch.long), caches
    )
    for row, (exp_token, exp_len) in enumerate(zip(expected_tokens, expected_lens)):
        assert int(batched_logits[row].argmax().item()) == exp_token
        assert cache_seq_len(batched_caches[row]) == exp_len


def test_qwen_continuous_greedy_matches_solo(qwen_pair) -> None:
    _, cached, _, tokenizer = qwen_pair
    params = SamplingParams(max_tokens=6, temperature=0.0)
    prompts = ["Hi", "The capital of France is"]
    solos = [cached.generate(text, params) for text in prompts]

    async def main() -> list[Sequence]:
        worker = ContinuousWorker(cached, Scheduler(RequestQueue(16), max_batch=8))
        seqs = [
            Sequence(prompt_ids=encode_text(tokenizer, text), sampling=params) for text in prompts
        ]
        task = asyncio.create_task(worker.run())
        try:
            for seq in seqs:
                worker.scheduler.submit(seq)
            await asyncio.gather(*[seq.finished_event.wait() for seq in seqs])
        finally:
            worker.request_stop()
            await asyncio.wait_for(task, timeout=30)
        return seqs

    seqs = asyncio.run(main())
    for seq, solo in zip(seqs, solos):
        assert seq.result is not None
        assert seq.result.output_token_ids == solo.output_token_ids
        assert seq.result.finish_reason == solo.finish_reason


def test_qwen_short_finishes_before_long(qwen_pair) -> None:
    _, cached, _, tokenizer = qwen_pair

    async def main() -> tuple[Sequence, Sequence]:
        worker = ContinuousWorker(cached, Scheduler(RequestQueue(16), max_batch=8))
        short = Sequence(
            prompt_ids=encode_text(tokenizer, "Hi"),
            sampling=SamplingParams(max_tokens=2, temperature=0.0),
        )
        long = Sequence(
            prompt_ids=encode_text(tokenizer, "The capital of France is"),
            sampling=SamplingParams(max_tokens=10, temperature=0.0),
        )
        task = asyncio.create_task(worker.run())
        try:
            worker.scheduler.submit(short)
            worker.scheduler.submit(long)
            await asyncio.gather(short.finished_event.wait(), long.finished_event.wait())
        finally:
            worker.request_stop()
            await asyncio.wait_for(task, timeout=60)
        return short, long

    short, long = asyncio.run(main())
    assert short.finished_at is not None
    assert long.finished_at is not None
    assert short.finished_at < long.finished_at
    assert len(short.output_ids) <= 2
    assert len(long.output_ids) <= 10


def test_qwen_first_token_streams_before_finish(qwen_pair) -> None:
    _, cached, _, tokenizer = qwen_pair

    async def main() -> None:
        worker = ContinuousWorker(cached, Scheduler(RequestQueue(8), max_batch=8))
        seq = Sequence(
            prompt_ids=encode_text(tokenizer, "Hi"),
            sampling=SamplingParams(max_tokens=4, temperature=0.0),
        )
        task = asyncio.create_task(worker.run())
        try:
            worker.scheduler.submit(seq)
            token = await asyncio.wait_for(seq.token_queue.get(), timeout=15)
            assert token is not None
            assert not seq.finished_event.is_set()
            await seq.finished_event.wait()
            assert seq.result is not None
        finally:
            worker.request_stop()
            await asyncio.wait_for(task, timeout=30)

    asyncio.run(main())


def test_qwen_block_pool_freed(qwen_pair) -> None:
    from flux.engine.block_pool import BlockPool

    _, cached, _, tokenizer = qwen_pair
    pool = BlockPool(8, 16)

    async def main() -> None:
        worker = ContinuousWorker(cached, Scheduler(RequestQueue(8), max_batch=8, pool=pool))
        seq = Sequence(
            prompt_ids=encode_text(tokenizer, "The capital of France is"),
            sampling=SamplingParams(max_tokens=4, temperature=0.0),
        )
        task = asyncio.create_task(worker.run())
        try:
            worker.scheduler.submit(seq)
            await seq.finished_event.wait()
        finally:
            worker.request_stop()
            await asyncio.wait_for(task, timeout=30)

    asyncio.run(main())
    assert pool.used_blocks == 0


def test_qwen_prefill_from_prefix_matches_full(qwen_pair) -> None:
    from flux.engine.kv_utils import slice_cache

    _, cached, _, tokenizer = qwen_pair
    ids = encode_text(tokenizer, "Prefix reuse check for Qwen.", device="cpu")
    assert ids.shape[1] >= 8
    full_logits, full_cache = cached.prefill(ids)
    split = ids.shape[1] // 2
    prefix = slice_cache(full_cache, split)
    suffix_logits, suffix_cache = cached.prefill_from_prefix(ids[0, split:].tolist(), prefix)
    assert int(suffix_logits.argmax()) == int(full_logits.argmax())
    assert cache_seq_len(suffix_cache) == ids.shape[1]
