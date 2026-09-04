import asyncio

import torch

from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.scheduler import RequestQueue, Scheduler
from flux.engine.sequence import Sequence
from flux.engine.tokenizer import encode_text
from flux.engine.types import SamplingParams
from flux.engine.worker import ContinuousWorker, QueuedWorker


def _engine() -> CachedEngine:
    return CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")


def _seq(engine: CachedEngine, text: str, max_tokens: int) -> Sequence:
    ids = encode_text(engine.tokenizer, text, device=engine.device)
    return Sequence(prompt_ids=ids, sampling=SamplingParams(max_tokens=max_tokens, temperature=0.0))


async def _run_worker(worker, seqs: list[Sequence]) -> None:
    task = asyncio.create_task(worker.run())
    try:
        for seq in seqs:
            worker.scheduler.submit(seq)
        await asyncio.gather(*[seq.finished_event.wait() for seq in seqs])
    finally:
        worker.request_stop()
        await asyncio.wait_for(task, timeout=5)


def test_queued_worker_matches_solo_greedy() -> None:
    async def main() -> None:
        engine = _engine()
        worker = QueuedWorker(engine, Scheduler(RequestQueue(16), max_batch=8))
        seq = _seq(engine, "hello queue", max_tokens=6)
        await _run_worker(worker, [seq])
        solo = engine.generate("hello queue", seq.sampling)
        assert seq.result is not None
        assert seq.result.output_token_ids == solo.output_token_ids
        assert seq.output_ids == solo.output_token_ids
        assert seq.result.engine == "queued"

    asyncio.run(main())


def test_continuous_greedy_matches_solo() -> None:
    async def main() -> None:
        engine = _engine()
        worker = ContinuousWorker(engine, Scheduler(RequestQueue(16), max_batch=8))
        texts = ["alpha", "bravo batch", "charlie"]
        seqs = [_seq(engine, text, max_tokens=8) for text in texts]
        await _run_worker(worker, seqs)
        for seq, text in zip(seqs, texts):
            solo = engine.generate(text, seq.sampling)
            assert seq.result is not None
            assert seq.result.output_token_ids == solo.output_token_ids
            assert seq.result.engine == "continuous"

    asyncio.run(main())


def test_short_request_finishes_while_long_continues() -> None:
    async def main() -> None:
        engine = _engine()
        worker = ContinuousWorker(engine, Scheduler(RequestQueue(16), max_batch=8))
        short = _seq(engine, "short", max_tokens=2)
        long = _seq(engine, "this is a longer prompt for the long job", max_tokens=12)
        await _run_worker(worker, [short, long])
        assert short.finished_at is not None
        assert long.finished_at is not None
        assert short.finished_at < long.finished_at
        assert len(short.output_ids) == 2
        assert len(long.output_ids) == 12

    asyncio.run(main())


def test_max_batch_enforced() -> None:
    async def main() -> None:
        engine = _engine()
        sizes: list[int] = []
        original = engine.decode_batch

        def wrapped(tokens: torch.Tensor, caches):
            sizes.append(int(tokens.shape[0]))
            return original(tokens, caches)

        engine.decode_batch = wrapped  # type: ignore[method-assign]
        worker = ContinuousWorker(engine, Scheduler(RequestQueue(32), max_batch=8))
        seqs = [_seq(engine, f"req {i}", max_tokens=4) for i in range(10)]
        await _run_worker(worker, seqs)
        assert sizes
        assert max(sizes) <= 8
        assert 8 in sizes
        assert all(seq.result is not None for seq in seqs)

    asyncio.run(main())


def test_decode_batch_shape_and_greedy_match() -> None:
    model = FakeLM()
    engine = CachedEngine(model, FakeTokenizer(), device="cpu")
    prompts = [
        torch.tensor([[1, 4, 7]], dtype=torch.long),
        torch.tensor([[2, 3]], dtype=torch.long),
    ]
    caches = []
    last_tokens = []
    solos = []
    for prompt in prompts:
        logits, cache = engine.prefill(prompt)
        token = logits.argmax(dim=-1)
        solos.append((token, cache, prompt))
        caches.append(cache)
        last_tokens.append(int(token.item()))
    batched_logits, batched_caches = engine.decode_batch(
        torch.tensor(last_tokens, dtype=torch.long), caches
    )
    assert batched_logits.shape[0] == 2
    decode_calls = [c for c in model.calls if c["input_ids_shape"][0] == 2]
    assert decode_calls
    assert decode_calls[0]["input_ids_shape"] == (2, 1)
    for row, (token, cache, prompt) in enumerate(solos):
        # Fresh engine path: decode the same last token solo.
        solo_engine = CachedEngine(FakeLM(), FakeTokenizer(), device="cpu")
        solo_logits, solo_cache = solo_engine.prefill(prompt)
        solo_token = solo_logits.argmax(dim=-1)
        solo_next, solo_new = solo_engine.decode(solo_token, solo_cache)
        assert int(batched_logits[row].argmax().item()) == int(solo_next.argmax().item())
        assert batched_caches[row][0][0].shape[-2] == solo_new[0][0].shape[-2]
