from flux.config import Settings
from flux.engine.block_pool import BlockPool, QWEN_BYTES_PER_TOKEN, resolve_num_kv_blocks
from flux.engine.sequence import Sequence
from flux.engine.types import SamplingParams
import torch


def _seq() -> Sequence:
    return Sequence(prompt_ids=torch.ones(1, 8, dtype=torch.long), sampling=SamplingParams(max_tokens=8))


def test_allocate_and_free() -> None:
    pool = BlockPool(num_blocks=4, block_size=16)
    seq = _seq()
    ids = pool.allocate(seq, num_tokens=20)
    assert ids is not None
    assert len(ids) == 2
    assert pool.used_blocks == 2
    assert seq.block_ids == ids
    pool.free(seq)
    assert pool.used_blocks == 0
    assert seq.block_ids == []
    pool.free(seq)  # idempotent
    assert pool.used_blocks == 0


def test_allocate_returns_none_when_full() -> None:
    pool = BlockPool(num_blocks=1, block_size=16)
    first = _seq()
    assert pool.allocate(first, 16) is not None
    second = _seq()
    assert pool.allocate(second, 16) is None
    assert pool.used_blocks == 1


def test_blocks_needed() -> None:
    pool = BlockPool(num_blocks=8, block_size=16)
    assert pool.blocks_needed(0) == 0
    assert pool.blocks_needed(1) == 1
    assert pool.blocks_needed(16) == 1
    assert pool.blocks_needed(17) == 2


def test_resolve_explicit_and_auto_cap() -> None:
    explicit = Settings(num_kv_blocks="7", block_size=16, max_batch_size=8, max_seq_len=1024)
    assert resolve_num_kv_blocks(explicit) == 7
    auto = Settings(num_kv_blocks="auto", block_size=16, max_batch_size=2, max_seq_len=32)
    # cap = ceil(2 * 32 / 16) = 4, even if RAM budget is huge
    n = resolve_num_kv_blocks(auto, ram_bytes=64 * 1024**3, bytes_per_token=QWEN_BYTES_PER_TOKEN)
    assert n == 4
