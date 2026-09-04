"""Deterministic stand-in for a causal LM so CI never downloads Qwen."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FakeLMOutput:
    logits: torch.Tensor


class FakeLM(nn.Module):
    """Next token is last_id + 1 (wraps to 1). Token 0 is EOS and is never sampled."""

    def __init__(
        self,
        vocab_size: int = 32,
        n_layers: int = 2,
        n_kv_heads: int = 2,
        head_dim: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.eos_token_id = 0
        # Tiny parameter so .eval() / .to() behave like a real module.
        self.embed = nn.Embedding(vocab_size, 8)
        self.calls: list[dict] = []

    def forward(self, input_ids: torch.Tensor, **kwargs) -> FakeLMOutput:
        self.calls.append({"input_ids_shape": tuple(input_ids.shape), "kwargs": dict(kwargs)})
        batch, seq = input_ids.shape
        last = input_ids[:, -1]
        next_ids = last + 1
        next_ids = torch.where(next_ids >= self.vocab_size, torch.ones_like(next_ids), next_ids)
        next_ids = torch.where(next_ids == self.eos_token_id, torch.ones_like(next_ids), next_ids)
        logits = torch.zeros(batch, seq, self.vocab_size, dtype=torch.float32)
        logits[:, -1, :].fill_(-1e4)
        for row in range(batch):
            logits[row, -1, int(next_ids[row].item())] = 10.0
        return FakeLMOutput(logits=logits)


class FakeTokenizer:
    def __init__(self, vocab_size: int = 32) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.vocab_size = vocab_size

    def __call__(self, text: str, return_tensors: str | None = None, **_kwargs):
        ids = self.encode(text)
        tensor = torch.tensor([ids], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": tensor}
        return {"input_ids": ids}

    def encode(self, text: str, **_kwargs) -> list[int]:
        if not text:
            return [1]
        ids = [1]
        for char in text:
            ids.append((ord(char) % (self.vocab_size - 2)) + 1)
        return ids[:16]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars: list[str] = []
        for token in ids:
            token = int(token)
            if skip_special_tokens and token == self.eos_token_id:
                continue
            chars.append(chr(ord("a") + (token % 26)))
        return "".join(chars)
