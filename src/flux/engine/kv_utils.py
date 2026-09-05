"""Helpers for HF DynamicCache and FakeLM list-of-(K,V) caches."""

from __future__ import annotations

from typing import Any


def iter_kv(cache: Any) -> list[tuple[Any, Any]]:
    if cache is None:
        return []
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer[0], layer[1]) for layer in cache]


def cache_seq_len(cache: Any) -> int:
    layers = iter_kv(cache)
    if not layers:
        return 0
    key = layers[0][0]
    # HF / FakeLM: [batch, n_kv_heads, seq, head_dim]
    return int(key.shape[-2])


def cache_layer_shapes(cache: Any) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [(tuple(k.shape), tuple(v.shape)) for k, v in iter_kv(cache)]


def slice_cache(cache: Any, n_tokens: int) -> list[tuple[Any, Any]]:
    """Copy the first `n_tokens` positions into a new list-of-(K, V) cache."""
    if n_tokens < 0:
        raise ValueError("n_tokens must be >= 0")
    out: list[tuple[Any, Any]] = []
    for key, value in iter_kv(cache):
        out.append(
            (
                key[..., :n_tokens, :].contiguous().clone(),
                value[..., :n_tokens, :].contiguous().clone(),
            )
        )
    return out


def clone_cache(cache: Any) -> list[tuple[Any, Any]]:
    return slice_cache(cache, cache_seq_len(cache))


def describe_cache(cache: Any) -> dict[str, Any]:
    shapes = cache_layer_shapes(cache)
    if not shapes:
        return {"n_layers": 0, "seq_len": 0}
    k_shape, _ = shapes[0]
    return {
        "n_layers": len(shapes),
        "n_kv_heads": int(k_shape[1]) if len(k_shape) >= 4 else None,
        "seq_len": int(k_shape[-2]),
        "head_dim": int(k_shape[-1]) if len(k_shape) >= 2 else None,
        "k_shape": k_shape,
    }
