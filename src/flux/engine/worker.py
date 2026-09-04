"""Phase 4 sequential worker and Phase 5 continuous-batching loop."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import torch

from flux.engine.cached_engine import CachedEngine
from flux.engine.naive_engine import pack_result
from flux.engine.sampler import sample_logits
from flux.engine.scheduler import Scheduler
from flux.engine.sequence import Sequence, SequenceStatus
from flux.engine.tokenizer import combined_stop_ids

logger = logging.getLogger(__name__)


class WorkerStats:
    def __init__(self) -> None:
        self.last_batch_size = 0
        self.tokens_generated = 0
        self.running: list[Sequence] = []


def _queue_put(seq: Sequence, item: int | None) -> None:
    def _put() -> None:
        try:
            seq.token_queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    loop = seq._loop
    if loop is not None:
        loop.call_soon_threadsafe(_put)
    else:
        _put()


def _signal_end(seq: Sequence) -> None:
    _queue_put(seq, None)
    seq.finished_event.set()


def _release(seq: Sequence, scheduler: Scheduler) -> None:
    if scheduler.pool is not None:
        scheduler.pool.free(seq)
    seq.kv_cache = None


def _abort_one(seq: Sequence, scheduler: Scheduler) -> None:
    seq.status = SequenceStatus.ABORTED
    seq.finished_at = time.perf_counter()
    _release(seq, scheduler)
    _signal_end(seq)


def _abort_leftovers(scheduler: Scheduler, stats: WorkerStats) -> None:
    while True:
        seq = scheduler.queue.pop_one()
        if seq is None:
            break
        _abort_one(seq, scheduler)
    for seq in stats.running:
        if not seq.finished_event.is_set():
            _abort_one(seq, scheduler)
    stats.running = []


def _finish_sequence(seq: Sequence, tokenizer: Any, engine_name: str, scheduler: Scheduler) -> None:
    reason = seq.finish_reason or "length"
    if seq.started is None:
        seq.started = time.perf_counter()
    seq.result = pack_result(
        tokenizer=tokenizer,
        prompt_token_ids=seq.prompt_token_ids,
        output_ids=seq.output_ids,
        finish_reason=reason,
        started=seq.started,
        first_token_at=seq.first_token_at,
        engine=engine_name,
    )
    seq.status = SequenceStatus.FINISHED
    seq.finished_at = time.perf_counter()
    _release(seq, scheduler)
    _signal_end(seq)


def _note_token(seq: Sequence, token: int, stop_ids: set[int]) -> None:
    seq.output_ids.append(token)
    seq.last_token = token
    if seq.first_token_at is None:
        seq.first_token_at = time.perf_counter()
    _queue_put(seq, token)
    if token in stop_ids:
        seq.finish_reason = "stop"
    elif len(seq.output_ids) >= seq.sampling.max_tokens:
        seq.finish_reason = "length"


def _wants_abort(seq: Sequence) -> bool:
    return seq.abort_requested or seq.status == SequenceStatus.ABORTED


class QueuedWorker:
    """Phase 4: one sequence at a time from the waiting queue."""

    def __init__(self, engine: CachedEngine, scheduler: Scheduler) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.stats = WorkerStats()
        self.engine_name = "queued"
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()
        self.scheduler.queue.wake()

    async def run(self) -> None:
        logger.info("queued worker started")
        while not self._stop.is_set():
            if len(self.scheduler.queue) == 0:
                await self.scheduler.queue.wait_not_empty()
            if self._stop.is_set():
                break
            # Room of 1 even if the scheduler's max_batch is larger.
            hold = max(0, self.scheduler.max_batch - 1)
            admitted = self.scheduler.admit(hold)
            if not admitted:
                if self.scheduler.queue.peek_one() is not None:
                    await asyncio.sleep(0)
                continue
            seq = admitted[0]
            if _wants_abort(seq):
                _abort_one(seq, self.scheduler)
                continue
            seq.status = SequenceStatus.DECODING
            seq.started = time.perf_counter()
            self.stats.running = [seq]
            self.stats.last_batch_size = 1
            try:
                await asyncio.to_thread(self._run_one, seq)
            except Exception as exc:  # noqa: BLE001 — worker must not die on one request
                logger.exception("queued worker failed seq=%s", seq.id)
                seq.error = exc
                seq.status = SequenceStatus.ERROR
                seq.finished_at = time.perf_counter()
                _release(seq, self.scheduler)
                _signal_end(seq)
            finally:
                self.stats.running = []
        _abort_leftovers(self.scheduler, self.stats)
        logger.info("queued worker stopped")

    def _run_one(self, seq: Sequence) -> None:
        stop_ids = combined_stop_ids(
            self.engine.tokenizer, seq.sampling.stop_token_ids, seq.sampling.ignore_eos
        )
        logits, cache = self.engine.prefill(seq.prompt_ids)
        next_id = sample_logits(logits, seq.sampling)
        token = int(next_id.item())
        _note_token(seq, token, stop_ids)
        seq.kv_cache = cache
        self.stats.tokens_generated += 1
        if _wants_abort(seq):
            _abort_one(seq, self.scheduler)
            return
        if seq.finish_reason:
            _finish_sequence(seq, self.engine.tokenizer, self.engine_name, self.scheduler)
            return
        last_token = next_id
        while not seq.finish_reason:
            if _wants_abort(seq):
                _abort_one(seq, self.scheduler)
                return
            logits, cache = self.engine.decode(last_token, cache)
            next_id = sample_logits(logits, seq.sampling)
            token = int(next_id.item())
            _note_token(seq, token, stop_ids)
            seq.kv_cache = cache
            self.stats.tokens_generated += 1
            last_token = next_id
        _finish_sequence(seq, self.engine.tokenizer, self.engine_name, self.scheduler)


class ContinuousWorker:
    """Phase 5: admit at iteration boundaries, prefill one-at-a-time, batch decode."""

    def __init__(self, engine: CachedEngine, scheduler: Scheduler) -> None:
        self.engine = engine
        self.scheduler = scheduler
        self.stats = WorkerStats()
        self.engine_name = "continuous"
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()
        self.scheduler.queue.wake()

    async def run(self) -> None:
        logger.info("continuous worker started max_batch=%d", self.scheduler.max_batch)
        while not self._stop.is_set():
            if not self.stats.running and len(self.scheduler.queue) == 0:
                await self.scheduler.queue.wait_not_empty()
            if self._stop.is_set():
                break
            self._drop_aborted()
            admitted = self.scheduler.admit(len(self.stats.running))
            for seq in admitted:
                if _wants_abort(seq):
                    _abort_one(seq, self.scheduler)
                    continue
                await asyncio.to_thread(self._prefill_one, seq)
                if seq.status == SequenceStatus.FINISHED:
                    continue
                if seq.status == SequenceStatus.ABORTED:
                    continue
                if seq.status != SequenceStatus.ERROR:
                    self.stats.running.append(seq)
            if self.stats.running:
                await asyncio.to_thread(self._decode_one_step, self.stats.running)
                still: list[Sequence] = []
                for seq in self.stats.running:
                    if seq.status in {SequenceStatus.FINISHED, SequenceStatus.ERROR, SequenceStatus.ABORTED}:
                        continue
                    if _wants_abort(seq):
                        _abort_one(seq, self.scheduler)
                        continue
                    still.append(seq)
                self.stats.running = still
            elif not admitted:
                await asyncio.sleep(0)
        _abort_leftovers(self.scheduler, self.stats)
        logger.info("continuous worker stopped")

    def _drop_aborted(self) -> None:
        kept: list[Sequence] = []
        for seq in self.stats.running:
            if _wants_abort(seq):
                _abort_one(seq, self.scheduler)
                continue
            kept.append(seq)
        self.stats.running = kept

    def _prefill_one(self, seq: Sequence) -> None:
        if _wants_abort(seq):
            _abort_one(seq, self.scheduler)
            return
        seq.status = SequenceStatus.PREFILL
        seq.started = time.perf_counter()
        stop_ids = combined_stop_ids(
            self.engine.tokenizer, seq.sampling.stop_token_ids, seq.sampling.ignore_eos
        )
        try:
            logits, cache = self.engine.prefill(seq.prompt_ids)
            next_id = sample_logits(logits, seq.sampling)
            token = int(next_id.item())
            _note_token(seq, token, stop_ids)
            seq.kv_cache = cache
            self.stats.tokens_generated += 1
            if _wants_abort(seq):
                _abort_one(seq, self.scheduler)
                return
            if seq.finish_reason:
                logger.info(
                    "seq finished after prefill id=%s reason=%s new_tokens=%d",
                    seq.id,
                    seq.finish_reason,
                    len(seq.output_ids),
                )
                _finish_sequence(seq, self.engine.tokenizer, self.engine_name, self.scheduler)
            else:
                seq.status = SequenceStatus.DECODING
        except Exception as exc:  # noqa: BLE001
            logger.exception("prefill failed seq=%s", seq.id)
            seq.error = exc
            seq.status = SequenceStatus.ERROR
            seq.finished_at = time.perf_counter()
            _release(seq, self.scheduler)
            _signal_end(seq)

    def _decode_one_step(self, running: list[Sequence]) -> None:
        for seq in running:
            if _wants_abort(seq):
                _abort_one(seq, self.scheduler)
        live = [
            seq
            for seq in running
            if seq.status == SequenceStatus.DECODING
            and seq.last_token is not None
            and not _wants_abort(seq)
        ]
        if not live:
            return
        self.stats.last_batch_size = len(live)
        tokens = torch.tensor([seq.last_token for seq in live], dtype=torch.long, device=self.engine.device)
        caches = [seq.kv_cache for seq in live]
        try:
            logits, new_caches = self.engine.decode_batch(tokens, caches)
        except Exception as exc:  # noqa: BLE001
            logger.exception("decode_batch failed n=%d", len(live))
            for seq in live:
                seq.error = exc
                seq.status = SequenceStatus.ERROR
                seq.finished_at = time.perf_counter()
                _release(seq, self.scheduler)
                _signal_end(seq)
            return
        for row, seq in enumerate(live):
            if _wants_abort(seq):
                _abort_one(seq, self.scheduler)
                continue
            stop_ids = combined_stop_ids(
                self.engine.tokenizer, seq.sampling.stop_token_ids, seq.sampling.ignore_eos
            )
            next_id = sample_logits(logits[row], seq.sampling)
            token = int(next_id.item())
            _note_token(seq, token, stop_ids)
            seq.kv_cache = new_caches[row]
            self.stats.tokens_generated += 1
            if seq.finish_reason:
                logger.info(
                    "seq finished id=%s reason=%s new_tokens=%d running_left=%d",
                    seq.id,
                    seq.finish_reason,
                    len(seq.output_ids),
                    len(live) - 1,
                )
                _finish_sequence(seq, self.engine.tokenizer, self.engine_name, self.scheduler)
