import torch
import pytest

from flux.engine.scheduler import QueueFull, RequestQueue, Scheduler
from flux.engine.sequence import Sequence, SequenceStatus
from flux.engine.types import SamplingParams


def _seq(n: int = 1) -> Sequence:
    return Sequence(prompt_ids=torch.ones(1, n, dtype=torch.long), sampling=SamplingParams(max_tokens=4))


def test_queue_rejects_when_full() -> None:
    queue = RequestQueue(max_waiting=2)
    assert queue.try_enqueue(_seq())
    assert queue.try_enqueue(_seq())
    assert queue.try_enqueue(_seq()) is False
    assert len(queue) == 2


def test_scheduler_submit_raises_queue_full() -> None:
    scheduler = Scheduler(RequestQueue(max_waiting=1), max_batch=8)
    scheduler.submit(_seq())
    with pytest.raises(QueueFull):
        scheduler.submit(_seq())


def test_admit_is_fcfs_and_respects_max_batch() -> None:
    scheduler = Scheduler(RequestQueue(max_waiting=16), max_batch=3)
    seqs = [_seq() for _ in range(5)]
    for seq in seqs:
        scheduler.submit(seq)
    first = scheduler.admit(running_count=0)
    assert [s.id for s in first] == [s.id for s in seqs[:3]]
    assert len(scheduler.queue) == 2
    second = scheduler.admit(running_count=2)
    assert len(second) == 1
    assert second[0].id == seqs[3].id
    empty = scheduler.admit(running_count=3)
    assert empty == []
    leftover = scheduler.admit(running_count=0)
    assert [s.id for s in leftover] == [seqs[4].id]


def test_waiting_status_on_enqueue() -> None:
    seq = _seq()
    RequestQueue(4).try_enqueue(seq)
    assert seq.status == SequenceStatus.WAITING
