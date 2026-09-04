"""Pad/stack per-sequence KV caches into one decode batch and split them back."""

from __future__ import annotations

from typing import Any

import torch

from flux.engine.kv_utils import cache_seq_len, iter_kv


def stack_caches(caches: list[Any]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[int], int]:
    """Right-pad each cache to the max seq len and stack on the batch dim."""
    if not caches:
        raise ValueError("stack_caches requires at least one cache")
    lengths = [cache_seq_len(cache) for cache in caches]
    max_len = max(lengths)
    layers0 = iter_kv(caches[0])
    stacked: list[tuple[torch.Tensor, torch.Tensor]] = []
    batch = len(caches)
    for layer_idx, (k0, v0) in enumerate(layers0):
        _, heads, _, dim = k0.shape
        keys = k0.new_zeros((batch, heads, max_len, dim))
        values = v0.new_zeros((batch, heads, max_len, dim))
        for row, cache in enumerate(caches):
            key, value = iter_kv(cache)[layer_idx]
            seq_len = lengths[row]
            if seq_len:
                keys[row, :, :seq_len] = key[0, :, :seq_len]
                values[row, :, :seq_len] = value[0, :, :seq_len]
        stacked.append((keys, values))
    return stacked, lengths, max_len


def attention_mask_for_decode(lengths: list[int], max_past: int, device: torch.device) -> torch.Tensor:
    """`[B, max_past + 1]` mask: valid past tokens plus the new token at the end."""
    batch = len(lengths)
    mask = torch.zeros((batch, max_past + 1), dtype=torch.long, device=device)
    for row, seq_len in enumerate(lengths):
        if seq_len:
            mask[row, :seq_len] = 1
        mask[row, max_past] = 1
    return mask


def extract_row_cache(cache: Any, row: int, old_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """After batched decode, keep the original tokens plus the newly appended one."""
    extracted: list[tuple[torch.Tensor, torch.Tensor]] = []
    for key, value in iter_kv(cache):
        new_len = int(key.shape[-2])
        new_token = new_len - 1
        if old_len == 0:
            pieces_k = key[row : row + 1, :, new_token : new_token + 1]
            pieces_v = value[row : row + 1, :, new_token : new_token + 1]
        else:
            pieces_k = torch.cat(
                [
                    key[row : row + 1, :, :old_len],
                    key[row : row + 1, :, new_token : new_token + 1],
                ],
                dim=-2,
            )
            pieces_v = torch.cat(
                [
                    value[row : row + 1, :, :old_len],
                    value[row : row + 1, :, new_token : new_token + 1],
                ],
                dim=-2,
            )
        extracted.append((pieces_k.contiguous(), pieces_v.contiguous()))
    return extracted


def wrap_past(layers: list[tuple[torch.Tensor, torch.Tensor]], huggingface: bool) -> Any:
    if not huggingface:
        return layers
    from transformers import DynamicCache

    return DynamicCache.from_legacy_cache(tuple(layers))
