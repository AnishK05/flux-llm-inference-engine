"""Request waiting queue and FCFS admission."""

from __future__ import annotations

import asyncio
from collections import deque

from flux.engine.sequence import Sequence, SequenceStatus


class QueueFull(Exception):
    pass


class RequestQueue:
    def __init__(self, max_waiting: int) -> None:
        self.max_waiting = max_waiting
        self._waiting: deque[Sequence] = deque()
        self._event = asyncio.Event()

    def __len__(self) -> int:
        return len(self._waiting)

    def try_enqueue(self, seq: Sequence) -> bool:
        if len(self._waiting) >= self.max_waiting:
            return False
        seq.status = SequenceStatus.WAITING
        self._waiting.append(seq)
        self._event.set()
        return True

    def pop_one(self) -> Sequence | None:
        if not self._waiting:
            return None
        seq = self._waiting.popleft()
        if not self._waiting:
            self._event.clear()
        return seq

    def snapshot_ids(self) -> list[str]:
        return [seq.id for seq in self._waiting]

    def wake(self) -> None:
        """Unblock waiters (shutdown, or a producer that cannot enqueue)."""
        self._event.set()

    async def wait_not_empty(self) -> None:
        await self._event.wait()


class Scheduler:
    """FCFS admission: fill decode slots from the waiting queue."""

    def __init__(self, queue: RequestQueue, max_batch: int) -> None:
        self.queue = queue
        self.max_batch = max_batch

    def admit(self, running_count: int) -> list[Sequence]:
        room = max(0, self.max_batch - running_count)
        admitted: list[Sequence] = []
        while room > 0:
            seq = self.queue.pop_one()
            if seq is None:
                break
            admitted.append(seq)
            room -= 1
        return admitted

    def submit(self, seq: Sequence) -> None:
        if not self.queue.try_enqueue(seq):
            raise QueueFull(f"waiting queue is full ({self.queue.max_waiting})")
