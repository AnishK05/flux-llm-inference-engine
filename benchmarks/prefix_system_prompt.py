"""Tiny FakeLM walk of shared-system-prompt prefix reuse (CI-safe).

Does not publish official resume numbers. Run on Qwen separately if you want TTFT.
"""

from __future__ import annotations

from flux.engine.block_pool import BlockPool
from flux.engine.cached_engine import CachedEngine
from flux.engine.fake_lm import FakeLM, FakeTokenizer
from flux.engine.prefix_cache import PrefixCache
from flux.engine.scheduler import RequestQueue, Scheduler
from flux.engine.sequence import Sequence
from flux.engine.tokenizer import encode_text
from flux.engine.types import SamplingParams
from flux.engine.worker import ContinuousWorker
import asyncio


SYSTEM = "0123456789abcde"


def main() -> None:
    model = FakeLM()
    engine = CachedEngine(model, FakeTokenizer())
    pool = BlockPool(64, 16)
    prefix = PrefixCache(pool, block_size=16)
    scheduler = Scheduler(RequestQueue(8), max_batch=8, pool=pool, prefix_cache=prefix)

    async def run() -> None:
        worker = ContinuousWorker(engine, scheduler)
        task = asyncio.create_task(worker.run())
        try:
            for user in ("AAA", "BBB", "CCC"):
                ids = encode_text(engine.tokenizer, SYSTEM + user)
                seq = Sequence(prompt_ids=ids, sampling=SamplingParams(max_tokens=3, temperature=0.0))
                scheduler.submit(seq)
                await seq.finished_event.wait()
        finally:
            worker.request_stop()
            await task

    asyncio.run(run())
    print(prefix.snapshot())
    suffix = [c for c in model.calls if c.get("past_len", 0) > 0 and c["input_ids_shape"][1] > 1]
    print({"suffix_prefills": len(suffix), "calls": len(model.calls)})


if __name__ == "__main__":
    main()
