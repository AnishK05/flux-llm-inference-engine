import asyncio

import torch

from flux.engine.block_pool import BlockPool
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.kv_utils import cache_seq_len, clone_cache, slice_cache
from flux.engine.prefix_cache import PrefixCache
from flux.engine.scheduler import RequestQueue, Scheduler
from flux.engine.sequence import Sequence, SequenceStatus
from flux.engine.tokenizer import encode_text
from flux.engine.types import SamplingParams
from flux.engine.worker import ContinuousWorker


SHARED = "0123456789abcde"  # FakeTokenizer: 1 + 15 chars = 16 tokens


def _engine() -> CachedEngine:
    return CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")


def _ids(engine: CachedEngine, text: str) -> list[int]:
    return encode_text(engine.tokenizer, text, device=engine.device)[0].tolist()


def _seq(engine: CachedEngine, text: str, max_tokens: int = 4) -> Sequence:
    ids = encode_text(engine.tokenizer, text, device=engine.device)
    return Sequence(prompt_ids=ids, sampling=SamplingParams(max_tokens=max_tokens, temperature=0.0))


def test_slice_and_clone_are_copies() -> None:
    model = FakeLM()
    engine = CachedEngine(model, FakeTokenizer())
    _, cache = engine.prefill(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long))
    sliced = slice_cache(cache, 4)
    cloned = clone_cache(sliced)
    assert cache_seq_len(sliced) == 4
    assert cache_seq_len(cloned) == 4
    sliced[0][0].fill_(3.0)
    assert float(cloned[0][0][0, 0, 0, 0]) == 0.0


def test_longest_prefix_lookup() -> None:
    pool = BlockPool(32, 16)
    cache = PrefixCache(pool, min_tokens=8, block_size=16)
    engine = _engine()
    long = _ids(engine, SHARED + SHARED)  # 31 tokens
    _, kv = engine.prefill(torch.tensor([long], dtype=torch.long))
    logits = torch.zeros(1, engine.model.vocab_size)
    cache.insert(long, kv, logits, [])
    query = long + [7, 7, 7]
    hit = cache.peek(query)
    assert hit is not None
    assert hit.n_tokens == len(long)
    short = cache.peek(long[:16])
    assert short is not None
    assert short.n_tokens == 16


def test_prefill_from_prefix_matches_full_and_skips_prefix() -> None:
    model = FakeLM()
    engine = CachedEngine(model, FakeTokenizer())
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    full_logits, full_cache = engine.prefill(prompt)
    prefix = slice_cache(full_cache, 4)
    model.calls.clear()
    suffix_logits, suffix_cache = engine.prefill_from_prefix(prompt[0, 4:].tolist(), prefix)
    assert int(suffix_logits.argmax()) == int(full_logits.argmax())
    assert cache_seq_len(suffix_cache) == 8
    assert model.calls[0]["past_len"] == 4
    assert model.calls[0]["input_ids_shape"] == (1, 4)


async def _run(worker, seqs: list[Sequence]) -> None:
    task = asyncio.create_task(worker.run())
    try:
        for seq in seqs:
            worker.scheduler.submit(seq)
        await asyncio.gather(*[seq.finished_event.wait() for seq in seqs])
    finally:
        worker.request_stop()
        await asyncio.wait_for(task, timeout=5)


def test_second_request_prefills_only_suffix() -> None:
    async def main() -> None:
        model = FakeLM()
        engine = CachedEngine(model, FakeTokenizer(), device="cpu")
        pool = BlockPool(32, 16)
        prefix = PrefixCache(pool, min_tokens=8, block_size=16)
        worker = ContinuousWorker(
            engine, Scheduler(RequestQueue(16), max_batch=8, pool=pool, prefix_cache=prefix)
        )
        scheduler = worker.scheduler
        first = _seq(engine, SHARED + "AAA", max_tokens=3)
        second = _seq(engine, SHARED + "BBB", max_tokens=3)
        await _run(worker, [first])
        model.calls.clear()
        worker = ContinuousWorker(engine, scheduler)
        await _run(worker, [second])
        prefills = [c for c in model.calls if c["input_ids_shape"][1] > 1 or c["past_len"] > 0]
        suffix_calls = [c for c in model.calls if c["past_len"] == 16]
        assert suffix_calls
        assert suffix_calls[0]["input_ids_shape"][1] == len(second.prompt_token_ids) - 16
        assert all(c["input_ids_shape"][1] < len(second.prompt_token_ids) or c["past_len"] > 0 for c in prefills)
        assert prefix.hits >= 1
        assert pool.used_blocks == 0

    asyncio.run(main())


def test_greedy_outputs_match_full_prefill() -> None:
    async def main() -> None:
        engine = _engine()
        pool = BlockPool(32, 16)
        prefix = PrefixCache(pool, min_tokens=8, block_size=16)
        worker = ContinuousWorker(
            engine, Scheduler(RequestQueue(16), max_batch=8, pool=pool, prefix_cache=prefix)
        )
        text = SHARED + "user"
        first = _seq(engine, text, max_tokens=6)
        second = _seq(engine, text, max_tokens=6)
        await _run(worker, [first, second])
        solo = engine.generate(text, first.sampling)
        assert first.result is not None and second.result is not None
        assert first.result.output_token_ids == solo.output_token_ids
        assert second.result.output_token_ids == solo.output_token_ids

    asyncio.run(main())


def test_exact_prompt_skips_prefill_via_stored_logits() -> None:
    async def main() -> None:
        model = FakeLM()
        engine = CachedEngine(model, FakeTokenizer(), device="cpu")
        pool = BlockPool(32, 16)
        prefix = PrefixCache(pool, min_tokens=8, block_size=16)
        worker = ContinuousWorker(
            engine, Scheduler(RequestQueue(16), max_batch=8, pool=pool, prefix_cache=prefix)
        )
        text = SHARED + "same"
        scheduler = worker.scheduler
        await _run(worker, [_seq(engine, text, max_tokens=3)])
        model.calls.clear()
        worker = ContinuousWorker(engine, scheduler)
        second = _seq(engine, text, max_tokens=3)
        await _run(worker, [second])
        prompt_len = len(second.prompt_token_ids)
        full_prefills = [c for c in model.calls if c["past_len"] == 0 and c["input_ids_shape"][1] == prompt_len]
        assert full_prefills == []
        assert prefix.hits >= 1
        assert second.result is not None

    asyncio.run(main())


def test_concurrent_shared_prefix_uses_fewer_blocks() -> None:
    async def main() -> None:
        engine = _engine()
        pool = BlockPool(64, 16)
        prefix = PrefixCache(pool, min_tokens=8, block_size=16)
        scheduler = Scheduler(RequestQueue(16), max_batch=8, pool=pool, prefix_cache=prefix)
        worker = ContinuousWorker(engine, scheduler)
        a = _seq(engine, SHARED + "AAA", max_tokens=8)
        b = _seq(engine, SHARED + "BBB", max_tokens=8)
        full_a = pool.blocks_needed(a.reservation_tokens())
        full_b = pool.blocks_needed(b.reservation_tokens())
        seen: list[int] = []
        original = engine.decode_batch

        def wrapped(tokens, caches):
            seen.append(pool.used_blocks)
            return original(tokens, caches)

        engine.decode_batch = wrapped  # type: ignore[method-assign]
        await _run(worker, [a, b])
        assert seen
        assert max(seen) < full_a + full_b
        assert pool.used_blocks == 0
        assert prefix.hits >= 1

    asyncio.run(main())


def test_abort_does_not_leak_prefix_blocks() -> None:
    async def main() -> None:
        engine = CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")
        pool = BlockPool(32, 16)
        prefix = PrefixCache(pool, min_tokens=8, block_size=16)
        worker = ContinuousWorker(
            engine, Scheduler(RequestQueue(8), max_batch=8, pool=pool, prefix_cache=prefix)
        )
        seq = _seq(engine, SHARED + "abort-me", max_tokens=12)
        task = asyncio.create_task(worker.run())
        try:
            worker.scheduler.submit(seq)
            token = await asyncio.wait_for(seq.token_queue.get(), timeout=2)
            assert token is not None
            assert pool.used_blocks >= 1
            seq.request_abort()
            await asyncio.wait_for(seq.finished_event.wait(), timeout=2)
            assert seq.status == SequenceStatus.ABORTED
            assert pool.used_blocks == 0
        finally:
            worker.request_stop()
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(main())


def test_worker_without_prefix_cache_still_runs() -> None:
    async def main() -> None:
        engine = _engine()
        worker = ContinuousWorker(engine, Scheduler(RequestQueue(8), max_batch=8))
        seq = _seq(engine, "hello", max_tokens=3)
        await _run(worker, [seq])
        assert seq.result is not None
        assert getattr(worker.scheduler, "prefix_cache", None) is None

    asyncio.run(main())
