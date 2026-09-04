from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from flux.engine.types import GenerateResult, SamplingParams


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass
class Sequence:
    prompt_ids: torch.Tensor
    sampling: SamplingParams
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    arrival_ns: int = field(default_factory=lambda: time.time_ns())
    output_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    kv_cache: Any = None
    last_token: int | None = None
    finish_reason: str | None = None
    result: GenerateResult | None = None
    error: BaseException | None = None
    started: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    finished_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def prompt_token_ids(self) -> list[int]:
        ids = self.prompt_ids
        if ids.dim() == 2:
            ids = ids[0]
        return [int(x) for x in ids.tolist()]

    def is_done(self) -> bool:
        if self.finish_reason is not None:
            return True
        return len(self.output_ids) >= self.sampling.max_tokens
