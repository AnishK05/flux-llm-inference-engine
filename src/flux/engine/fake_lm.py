"""Deterministic stand-in for a causal LM so CI never downloads Qwen."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FakeLMOutput:
    logits: torch.Tensor
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class FakeLM(nn.Module):
    """Next token is last_id + 1 (wraps to 1). Token 0 is EOS and is never sampled.

    When `use_cache=True`, returns a list-of-(K, V) cache with HF layout
    `[batch, n_kv_heads, seq, head_dim]`.
    """

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
        self.embed = nn.Embedding(vocab_size, 8)
        self.calls: list[dict] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> FakeLMOutput:
        batch, seq = input_ids.shape
        past_len = 0
        if past_key_values is not None:
            past_len = int(past_key_values[0][0].shape[-2])
        self.calls.append(
            {
                "input_ids_shape": tuple(input_ids.shape),
                "kwargs": dict(kwargs),
                "use_cache": use_cache,
                "past_len": past_len,
            }
        )
        last = input_ids[:, -1]
        next_ids = last + 1
        next_ids = torch.where(next_ids >= self.vocab_size, torch.ones_like(next_ids), next_ids)
        next_ids = torch.where(next_ids == self.eos_token_id, torch.ones_like(next_ids), next_ids)
        logits = torch.zeros(batch, seq, self.vocab_size, dtype=torch.float32)
        logits[:, -1, :].fill_(-1e4)
        for row in range(batch):
            logits[row, -1, int(next_ids[row].item())] = 10.0

        cache = None
        if use_cache:
            new_len = past_len + seq
            cache = []
            for layer in range(self.n_layers):
                key = torch.zeros(batch, self.n_kv_heads, new_len, self.head_dim)
                value = torch.zeros(batch, self.n_kv_heads, new_len, self.head_dim)
                if past_key_values is not None:
                    key[:, :, :past_len] = past_key_values[layer][0]
                    value[:, :, :past_len] = past_key_values[layer][1]
                cache.append((key, value))
        return FakeLMOutput(logits=logits, past_key_values=cache)


class FakeTokenizer:
    def __init__(self, vocab_size: int = 32) -> None:
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.unk_token_id = None
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
        return ids[:48]

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

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
        **_kwargs,
    ):
        parts: list[str] = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            parts.append(f"<|{role}|>{content}<|im_end|>")
        text = "".join(parts)
        if add_generation_prompt:
            text += "<|assistant|>"
        if not tokenize:
            return text
        ids = self.encode(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in {"<|im_end|>", "<|endoftext|>"}:
            return self.eos_token_id
        return -1
