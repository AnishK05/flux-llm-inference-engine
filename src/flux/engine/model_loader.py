"""Load Qwen2.5-0.5B-Instruct onto CPU in fp32."""

from __future__ import annotations

import logging
from typing import Any

import torch

from flux.config import Settings
from flux.device import resolve_device

logger = logging.getLogger(__name__)


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    return mapping[key]


def load_causal_lm(settings: Settings) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(settings.device)
    dtype = torch_dtype(settings.dtype)
    if device == "cpu" and dtype != torch.float32:
        logger.warning("CPU path forcing fp32 (asked for %s)", settings.dtype)
        dtype = torch.float32

    logger.info("loading tokenizer %s", settings.model)
    tokenizer = AutoTokenizer.from_pretrained(settings.model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("loading weights %s dtype=%s device=%s", settings.model, dtype, device)
    # Qwen2.5 enables sliding window in config; SDPA does not implement it and warns.
    # Eager is the honest CPU default until we own attention in a later phase.
    attn_implementation = "eager" if device == "cpu" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        settings.model,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
