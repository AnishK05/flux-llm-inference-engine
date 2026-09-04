import torch

from flux.engine.batching import attention_mask_for_decode, extract_row_cache, stack_caches
from flux.engine.kv_utils import cache_seq_len


def _cache(seq_len: int, batch: int = 1, heads: int = 2, dim: int = 4, layers: int = 2, fill: float = 1.0):
    out = []
    for layer in range(layers):
        key = torch.full((batch, heads, seq_len, dim), float(fill + layer), dtype=torch.float32)
        value = torch.full((batch, heads, seq_len, dim), float(fill + layer + 0.5), dtype=torch.float32)
        out.append((key, value))
    return out


def test_stack_right_pads_to_max_len() -> None:
    short = _cache(2, fill=1.0)
    long = _cache(5, fill=3.0)
    stacked, lengths, max_len = stack_caches([short, long])
    assert lengths == [2, 5]
    assert max_len == 5
    key0, _ = stacked[0]
    assert key0.shape == (2, 2, 5, 4)
    assert torch.equal(key0[0, :, :2], short[0][0][0])
    assert torch.all(key0[0, :, 2:] == 0)
    assert torch.equal(key0[1, :, :5], long[0][0][0])


def test_attention_mask_marks_pad_and_new_token() -> None:
    mask = attention_mask_for_decode([2, 5], max_past=5, device=torch.device("cpu"))
    assert mask.shape == (2, 6)
    assert mask[0].tolist() == [1, 1, 0, 0, 0, 1]
    assert mask[1].tolist() == [1, 1, 1, 1, 1, 1]


def test_extract_row_keeps_old_tokens_plus_appended() -> None:
    # Simulate right-padded seq=2 in a max_past=5 cache after appending at the end.
    padded = _cache(6, batch=2, fill=1.0)
    # Overwrite row 0 real tokens and the new token slot.
    padded[0][0][0, :, :2] = 7
    padded[0][0][0, :, 5] = 9
    extracted = extract_row_cache(padded, row=0, old_len=2)
    assert cache_seq_len(extracted) == 3
    key = extracted[0][0]
    assert torch.all(key[0, :, :2] == 7)
    assert torch.all(key[0, :, 2] == 9)
