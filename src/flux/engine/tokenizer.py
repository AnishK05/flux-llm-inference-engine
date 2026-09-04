"""Chat-template + encode/decode wrapper around a Hugging Face (or Fake) tokenizer."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

ChatTurn = Mapping[str, str]


def encode_text(tokenizer: Any, text: str, device: str = "cpu") -> torch.Tensor:
    if not text:
        raise ValueError("prompt must be non-empty")
    encoded = tokenizer(text, return_tensors="pt")
    return encoded["input_ids"].to(device)


def encode_chat(
    tokenizer: Any,
    messages: Sequence[ChatTurn],
    device: str = "cpu",
    add_generation_prompt: bool = True,
) -> torch.Tensor:
    if not messages:
        raise ValueError("messages must be non-empty")
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("tokenizer does not implement apply_chat_template")
    ids = tokenizer.apply_chat_template(
        list(messages),
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_tensors="pt",
    )
    if isinstance(ids, torch.Tensor):
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(device)
    return torch.tensor([list(ids)], dtype=torch.long, device=device)


def decode_tokens(tokenizer: Any, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)


def stop_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """EOS plus Qwen `<|im_end|>` when present. Deduped, stable order."""
    found: list[int] = []

    def _push(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _push(item)
            return
        token_id = int(value)
        if token_id < 0:
            return
        if token_id not in found:
            found.append(token_id)

    _push(getattr(tokenizer, "eos_token_id", None))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        for piece in ("<|im_end|>", "<|endoftext|>"):
            try:
                token_id = convert(piece)
            except Exception:
                continue
            if token_id is None:
                continue
            token_id = int(token_id)
            if unk is not None and token_id == int(unk):
                continue
            _push(token_id)
    return tuple(found)


def combined_stop_ids(tokenizer: Any, params_stop: Sequence[int], ignore_eos: bool) -> set[int]:
    extra = set(int(x) for x in params_stop)
    if ignore_eos:
        return extra
    return extra | set(stop_token_ids(tokenizer))
