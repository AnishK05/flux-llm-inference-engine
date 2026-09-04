"""Accounting-only KV block pool. Physical cache stays HF / padded tensors."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from flux.config import Settings
from flux.engine.sequence import Sequence

logger = logging.getLogger(__name__)

# Qwen2.5-0.5B fp32: 2 * layers * kv_heads * head_dim * 4 bytes.
QWEN_BYTES_PER_TOKEN = 2 * 24 * 2 * 64 * 4  # 24_576


class BlockPool:
    """Virtual-memory-style block allocator used for admission, not paging."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self._alloc: dict[str, list[int]] = {}

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return math.ceil(num_tokens / self.block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        return self.free_blocks >= self.blocks_needed(num_tokens)

    def allocate(self, seq: Sequence, num_tokens: int) -> list[int] | None:
        """Reserve blocks for the worst-case length of `seq`. None if it does not fit."""
        needed = self.blocks_needed(num_tokens)
        if needed == 0:
            self._alloc[seq.id] = []
            seq.block_ids = []
            return []
        if needed > self.free_blocks:
            return None
        ids = [self._free.popleft() for _ in range(needed)]
        self._alloc[seq.id] = ids
        seq.block_ids = list(ids)
        return ids

    def free(self, seq: Sequence) -> None:
        ids = self._alloc.pop(seq.id, None)
        if ids is None:
            ids = list(seq.block_ids)
        for block_id in ids:
            self._free.append(block_id)
        seq.block_ids = []

    def snapshot(self) -> dict[str, int]:
        return {
            "kv_blocks_used": self.used_blocks,
            "kv_blocks_free": self.free_blocks,
            "kv_blocks_total": self.num_blocks,
            "kv_block_size": self.block_size,
        }


def bytes_per_token_from_model(model: Any | None) -> int:
    config = getattr(model, "config", None)
    if config is None:
        return QWEN_BYTES_PER_TOKEN
    try:
        n_layers = int(config.num_hidden_layers)
        n_kv = int(config.num_key_value_heads)
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
        return 2 * n_layers * n_kv * head_dim * 4
    except Exception:
        return QWEN_BYTES_PER_TOKEN


def usable_ram_bytes() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            available = None
            total = None
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                elif line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
            if available is not None:
                return available
            if total is not None:
                return total
    except OSError:
        pass
    return 8 * 1024**3


def resolve_num_kv_blocks(
    settings: Settings,
    *,
    bytes_per_token: int | None = None,
    ram_bytes: int | None = None,
    model: Any | None = None,
) -> int:
    raw = str(settings.num_kv_blocks).strip().lower()
    if raw and raw != "auto":
        n = int(raw)
        if n < 1:
            raise ValueError("num_kv_blocks must be >= 1")
        return n

    bpt = bytes_per_token if bytes_per_token is not None else bytes_per_token_from_model(model)
    ram = ram_bytes if ram_bytes is not None else usable_ram_bytes()
    budget = 0.20 * ram
    block_bytes = settings.block_size * bpt
    from_ram = int(budget // block_bytes) if block_bytes else 1
    cap = math.ceil(settings.max_batch_size * settings.max_seq_len / settings.block_size)
    chosen = max(1, min(from_ram, cap))
    logger.info(
        "kv pool auto: ram=%d budget=%.0f bytes_per_token=%d block_size=%d "
        "block_bytes=%d from_ram=%d cap=%d -> num_blocks=%d",
        ram,
        budget,
        bpt,
        settings.block_size,
        block_bytes,
        from_ram,
        cap,
        chosen,
    )
    return chosen
