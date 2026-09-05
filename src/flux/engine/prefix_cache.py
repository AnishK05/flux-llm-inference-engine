"""Prefix / system-prompt KV reuse.

Identical leading token ids share block accounting while a request is live,
and clone stored prefix tensors so the next prefill only runs the suffix.
Decode still owns a private cache (HF DynamicCache grows in place).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from flux.engine.block_pool import BlockPool
from flux.engine.kv_utils import cache_seq_len, slice_cache
from flux.engine.sequence import Sequence

logger = logging.getLogger(__name__)


def _clone_logits(logits: Any) -> Any:
    if logits is None:
        return None
    if hasattr(logits, "detach"):
        return logits.detach().clone()
    return logits


@dataclass
class PrefixHit:
    token_ids: tuple[int, ...]
    n_tokens: int
    block_ids: list[int]
    kv_cache: Any
    last_logits: Any | None


@dataclass
class PrefixEntry:
    token_ids: tuple[int, ...]
    kv_cache: Any
    last_logits: Any | None
    block_ids: list[int] = field(default_factory=list)
    refs: int = 0
    hits: int = 0
    last_used: float = 0.0


class PrefixCache:
    def __init__(
        self,
        pool: BlockPool | None,
        *,
        max_entries: int = 64,
        min_tokens: int = 8,
        block_size: int = 16,
    ) -> None:
        self.pool = pool
        self.max_entries = max(1, max_entries)
        self.min_tokens = max(1, min_tokens)
        self.block_size = block_size
        self._entries: dict[tuple[int, ...], PrefixEntry] = {}
        self._block_pins: dict[int, int] = {}
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    def candidate_lengths(self, token_ids: list[int]) -> list[int]:
        n = len(token_ids)
        lengths: list[int] = []
        if n >= self.min_tokens:
            lengths.append(n)
        aligned = (n // self.block_size) * self.block_size
        while aligned >= self.min_tokens:
            if aligned not in lengths:
                lengths.append(aligned)
            aligned -= self.block_size
        return lengths

    def peek(self, token_ids: list[int]) -> PrefixHit | None:
        """Longest matching prefix without taking a ref or reserving blocks."""
        for n in self.candidate_lengths(token_ids):
            key = tuple(token_ids[:n])
            entry = self._entries.get(key)
            if entry is None:
                continue
            return PrefixHit(
                token_ids=key,
                n_tokens=n,
                block_ids=list(entry.block_ids),
                kv_cache=entry.kv_cache,
                last_logits=entry.last_logits if n == len(token_ids) else None,
            )
        return None

    def lookup(self, token_ids: list[int]) -> PrefixHit | None:
        hit = self.peek(token_ids)
        if hit is None:
            self.misses += 1
            return None
        entry = self._entries[hit.token_ids]
        self._acquire_blocks(entry)
        entry.hits += 1
        entry.last_used = time.time()
        self.hits += 1
        self.tokens_saved += len(entry.token_ids)
        return PrefixHit(
            token_ids=entry.token_ids,
            n_tokens=len(entry.token_ids),
            block_ids=list(entry.block_ids),
            kv_cache=entry.kv_cache,
            last_logits=entry.last_logits if len(entry.token_ids) == len(token_ids) else None,
        )

    def insert(
        self,
        token_ids: list[int],
        kv_cache: Any,
        last_logits: Any | None,
        seq_block_ids: list[int] | None = None,
    ) -> None:
        _ = seq_block_ids
        if kv_cache is None or cache_seq_len(kv_cache) < self.min_tokens:
            return
        now = time.time()
        stored_logits = _clone_logits(last_logits)
        for n in self.candidate_lengths(token_ids):
            if n > cache_seq_len(kv_cache):
                continue
            key = tuple(token_ids[:n])
            if key in self._entries:
                continue
            self._evict()
            if len(self._entries) >= self.max_entries:
                continue
            self._entries[key] = PrefixEntry(
                token_ids=key,
                kv_cache=slice_cache(kv_cache, n),
                last_logits=stored_logits if n == len(token_ids) else None,
                block_ids=[],
                refs=0,
                last_used=now,
            )
            logger.info("prefix cache insert tokens=%d entries=%d", n, len(self._entries))

    def adopt(self, seq: Sequence) -> None:
        """After a miss prefill, pin the stored prefix blocks to this live sequence."""
        if seq.prefix_key is not None:
            return
        ids = seq.prompt_token_ids
        lengths = self.candidate_lengths(ids)
        if not lengths:
            return
        key = tuple(ids[: lengths[0]])
        entry = self._entries.get(key)
        if entry is None:
            return
        if entry.refs > 0:
            self._acquire_blocks(entry)
            seq.prefix_key = key
            return
        n_blocks = self.pool.blocks_needed(len(entry.token_ids)) if self.pool is not None else 0
        taken = self.pool.detach(seq, n_blocks) if self.pool is not None else []
        entry.block_ids = taken
        self._pin(taken)
        entry.refs = 1
        entry.last_used = time.time()
        seq.prefix_key = key
        if self.pool is not None:
            seq.owned_block_ids = list(self.pool._alloc.get(seq.id, []))
        seq.block_ids = list(taken) + list(seq.owned_block_ids)

    def release(self, token_ids: tuple[int, ...] | None) -> None:
        if not token_ids:
            return
        entry = self._entries.get(token_ids)
        if entry is None:
            return
        if entry.refs <= 0:
            return
        self._unpin(entry.block_ids)
        entry.refs -= 1
        if entry.refs == 0:
            entry.block_ids = []

    def snapshot(self) -> dict[str, int]:
        live_refs = sum(entry.refs for entry in self._entries.values())
        return {
            "prefix_entries": len(self._entries),
            "prefix_hits": self.hits,
            "prefix_misses": self.misses,
            "prefix_tokens_saved": self.tokens_saved,
            "prefix_live_refs": live_refs,
        }

    def _acquire_blocks(self, entry: PrefixEntry) -> None:
        if self.pool is None:
            entry.refs += 1
            return
        if entry.refs > 0:
            self._pin(entry.block_ids)
            entry.refs += 1
            return
        shared = self._share_from_live(entry.token_ids)
        if shared is not None:
            entry.block_ids = shared
            self._pin(shared)
            entry.refs += 1
            return
        ids = self.pool.reserve(len(entry.token_ids))
        if ids is None:
            entry.block_ids = []
            entry.refs += 1
            return
        entry.block_ids = ids
        self._pin(ids)
        entry.refs += 1

    def _share_from_live(self, token_ids: tuple[int, ...]) -> list[int] | None:
        if self.pool is None:
            return None
        n_blocks = self.pool.blocks_needed(len(token_ids))
        for other in self._entries.values():
            if other.refs <= 0 or not other.block_ids:
                continue
            if other.token_ids[: len(token_ids)] == token_ids:
                return list(other.block_ids[:n_blocks])
        return None

    def _pin(self, ids: list[int]) -> None:
        for block_id in ids:
            self._block_pins[block_id] = self._block_pins.get(block_id, 0) + 1

    def _unpin(self, ids: list[int]) -> None:
        if self.pool is None:
            return
        to_free: list[int] = []
        for block_id in ids:
            left = self._block_pins.get(block_id, 0) - 1
            if left <= 0:
                self._block_pins.pop(block_id, None)
                to_free.append(block_id)
            else:
                self._block_pins[block_id] = left
        if to_free:
            self.pool.release(to_free)

    def _evict(self) -> None:
        while len(self._entries) >= self.max_entries:
            unused = [e for e in self._entries.values() if e.refs == 0]
            if not unused:
                return
            victim = min(unused, key=lambda e: e.last_used)
            self._unpin(victim.block_ids)
            self._entries.pop(victim.token_ids, None)
