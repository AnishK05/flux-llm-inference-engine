"""Phase 1 baseline: full-sequence forward every token, no KV cache.

This file stays importable forever. Later phases compare against it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from flux.engine.types import GenerateResult, SamplingParams

logger = logging.getLogger(__name__)


class NaiveEngine:
    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_loaded = True

    def generate(self, prompt: str, sampling: SamplingParams | None = None) -> GenerateResult:
        sampling = sampling or SamplingParams()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        if sampling.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        prompt_token_ids = input_ids[0].tolist()
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        stop_ids = set(sampling.stop_token_ids)
        if eos_id is not None and not sampling.ignore_eos:
            stop_ids.add(int(eos_id))

        output_ids: list[int] = []
        finish_reason = "length"
        started = time.perf_counter()

        with torch.inference_mode():
            for _ in range(sampling.max_tokens):
                outputs = self.model(input_ids, use_cache=False)
                logits = outputs.logits[:, -1, :]
                next_id = _sample(logits, sampling.temperature)
                token = int(next_id.item())
                output_ids.append(token)
                if token in stop_ids:
                    finish_reason = "stop"
                    break
                input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)

        latency_s = time.perf_counter() - started
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        completion_tokens = len(output_ids)
        tokens_per_second = completion_tokens / latency_s if latency_s > 0 else 0.0
        logger.info(
            "naive generate prompt_tokens=%d completion_tokens=%d latency_ms=%.1f tok_s=%.2f finish=%s",
            len(prompt_token_ids),
            completion_tokens,
            latency_s * 1000.0,
            tokens_per_second,
            finish_reason,
        )
        return GenerateResult(
            text=text,
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_ids,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            latency_s=latency_s,
            tokens_per_second=tokens_per_second,
        )


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
