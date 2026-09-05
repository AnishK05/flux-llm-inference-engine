"""Request waiting queue and FCFS / memory-fit admission."""

from __future__ import annotations

import asyncio
import logging
from collections import deque

from flux.engine.block_pool import BlockPool
from flux.engine.prefix_cache import PrefixCache, PrefixHit
from flux.engine.sequence import Sequence, SequenceStatus

logger = logging.getLogger(__name__)


class QueueFull(Exception):
    pass


class RequestTooLarge(Exception):
    """Request needs more KV blocks than the entire pool."""


class RequestQueue:
    def __init__(self, max_waiting: int) -> None:
        self.max_waiting = max_waiting
        self._waiting: deque[Sequence] = deque()
        self._event = asyncio.Event()

    def __len__(self) -> int:
        return len(self._waiting)

    def __iter__(self):
        return iter(self._waiting)

    def try_enqueue(self, seq: Sequence) -> bool:
        if len(self._waiting) >= self.max_waiting:
            return False
        seq.status = SequenceStatus.WAITING
        self._waiting.append(seq)
        self._event.set()
        return True

    def peek_one(self) -> Sequence | None:
        if not self._waiting:
            return None
        return self._waiting[0]

    def pop_one(self) -> Sequence | None:
        if not self._waiting:
            return None
        seq = self._waiting.popleft()
        if not self._waiting:
            self._event.clear()
        return seq

    def push_front(self, seq: Sequence) -> None:
        self._waiting.appendleft(seq)
        self._event.set()

    def remove(self, seq: Sequence) -> bool:
        kept: deque[Sequence] = deque()
        found = False
        for item in self._waiting:
            if not found and item is seq:
                found = True
                continue
            kept.append(item)
        self._waiting = kept
        if not self._waiting:
            self._event.clear()
        return found

    def snapshot_ids(self) -> list[str]:
        return [seq.id for seq in self._waiting]

    def wake(self) -> None:
        """Unblock waiters (shutdown, or a producer that cannot enqueue)."""
        self._event.set()

    async def wait_not_empty(self) -> None:
        await self._event.wait()


class Scheduler:
    """Admit waiting sequences into the decode batch.

    `fcfs`: head-of-line blocking when the oldest request does not fit the pool.
    `memory_fit`: skip waiters that do not fit and admit the next one that does.
    """

    def __init__(
        self,
        queue: RequestQueue,
        max_batch: int,
        pool: BlockPool | None = None,
        policy: str = "fcfs",
        prefix_cache: PrefixCache | None = None,
    ) -> None:
        self.queue = queue
        self.max_batch = max_batch
        self.pool = pool
        self.prefix_cache = prefix_cache
        name = (policy or "fcfs").strip().lower()
        if name not in {"fcfs", "memory_fit"}:
            logger.warning("unknown scheduler policy %r, using fcfs", policy)
            name = "fcfs"
        self.policy = name

    def _need_tokens(self, seq: Sequence) -> int:
        need = seq.reservation_tokens()
        if self.prefix_cache is None:
            return need
        hit = self.prefix_cache.peek(seq.prompt_token_ids)
        if hit is None:
            return need
        return max(0, need - hit.n_tokens)

    def _apply_hit(self, seq: Sequence, hit: PrefixHit) -> None:
        seq.prefix_key = hit.token_ids
        seq.prefix_tokens = hit.n_tokens
        seq.prefix_kv = hit.kv_cache
        seq.prefix_logits = hit.last_logits

    def submit(self, seq: Sequence) -> None:
        if self.pool is not None:
            needed = self.pool.blocks_needed(self._need_tokens(seq))
            if needed > self.pool.num_blocks:
                raise RequestTooLarge(
                    f"request needs {needed} KV blocks but pool has {self.pool.num_blocks}"
                )
        if not self.queue.try_enqueue(seq):
            raise QueueFull(f"waiting queue is full ({self.queue.max_waiting})")

    def admit(self, running_count: int) -> list[Sequence]:
        room = max(0, self.max_batch - running_count)
        if room <= 0:
            return []
        if self.policy == "memory_fit":
            return self._admit_memory_fit(room)
        return self._admit_fcfs(room)

    def _try_allocate(self, seq: Sequence) -> bool:
        hit = None
        if self.prefix_cache is not None:
            hit = self.prefix_cache.lookup(seq.prompt_token_ids)
        need = seq.reservation_tokens()
        if hit is not None:
            need = max(0, need - hit.n_tokens)
        if self.pool is None:
            if hit is not None:
                self._apply_hit(seq, hit)
            return True
        ids = self.pool.allocate(seq, need)
        if ids is None:
            if hit is not None and self.prefix_cache is not None:
                self.prefix_cache.release(hit.token_ids)
            return False
        seq.owned_block_ids = list(ids)
        if hit is not None:
            self._apply_hit(seq, hit)
            seq.block_ids = list(hit.block_ids) + list(ids)
        return True

    def _fits(self, seq: Sequence) -> bool:
        if self.pool is None:
            return True
        return self.pool.can_allocate(self._need_tokens(seq))

    def _admit_fcfs(self, room: int) -> list[Sequence]:
        admitted: list[Sequence] = []
        while room > 0:
            seq = self.queue.peek_one()
            if seq is None:
                break
            if not self._fits(seq):
                break
            self.queue.pop_one()
            if not self._try_allocate(seq):
                self.queue.push_front(seq)
                break
            admitted.append(seq)
            room -= 1
        return admitted

    def _admit_memory_fit(self, room: int) -> list[Sequence]:
        admitted: list[Sequence] = []
        while room > 0:
            chosen: Sequence | None = None
            for seq in self.queue:
                if self._fits(seq):
                    chosen = seq
                    break
            if chosen is None:
                break
            self.queue.remove(chosen)
            if not self._try_allocate(chosen):
                self.queue.push_front(chosen)
                break
            admitted.append(chosen)
            room -= 1
        return admitted
