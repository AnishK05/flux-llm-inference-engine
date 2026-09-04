"""Phase 1 baseline: full-sequence forward every token, no KV cache.

This file stays importable forever. Later phases compare against it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import torch

from flux.engine.sampler import sample_logits
from flux.engine.tokenizer import combined_stop_ids, decode_tokens, encode_chat, encode_text
from flux.engine.types import GenerateResult, SamplingParams

logger = logging.getLogger("flux.engine")


class NaiveEngine:
    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_loaded = True
        self.engine_name = "naive"

    def generate(self, prompt: str, sampling: SamplingParams | None = None) -> GenerateResult:
        sampling = sampling or SamplingParams()
        input_ids = encode_text(self.tokenizer, prompt, device=self.device)
        return self.generate_ids(input_ids, sampling)

    def generate_chat(
        self,
        messages: Sequence[dict[str, str]],
        sampling: SamplingParams | None = None,
    ) -> GenerateResult:
        sampling = sampling or SamplingParams()
        input_ids = encode_chat(self.tokenizer, messages, device=self.device)
        templated = decode_tokens(self.tokenizer, input_ids[0].tolist(), skip_special_tokens=False)
        logger.info("naive chat template: %s", templated[:500])
        return self.generate_ids(input_ids, sampling)

    def generate_ids(self, input_ids: torch.Tensor, sampling: SamplingParams) -> GenerateResult:
        if sampling.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        prompt_token_ids = input_ids[0].tolist()
        stop_ids = combined_stop_ids(self.tokenizer, sampling.stop_token_ids, sampling.ignore_eos)

        output_ids: list[int] = []
        finish_reason = "length"
        started = time.perf_counter()
        first_token_at: float | None = None

        with torch.inference_mode():
            for _ in range(sampling.max_tokens):
                outputs = self.model(input_ids, use_cache=False)
                logits = outputs.logits[:, -1, :]
                next_id = sample_logits(logits, sampling)
                token = int(next_id.item())
                output_ids.append(token)
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                if token in stop_ids:
                    finish_reason = "stop"
                    break
                input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)

        return pack_result(
            tokenizer=self.tokenizer,
            prompt_token_ids=prompt_token_ids,
            output_ids=output_ids,
            finish_reason=finish_reason,
            started=started,
            first_token_at=first_token_at,
            engine=self.engine_name,
        )


def pack_result(
    *,
    tokenizer: Any,
    prompt_token_ids: list[int],
    output_ids: list[int],
    finish_reason: str,
    started: float,
    first_token_at: float | None,
    engine: str,
) -> GenerateResult:
    ended = time.perf_counter()
    latency_s = ended - started
    ttft_s = (first_token_at - started) if first_token_at is not None else latency_s
    completion_tokens = len(output_ids)
    tpot_s = None
    if completion_tokens > 1:
        tpot_s = (ended - started - ttft_s) / (completion_tokens - 1)
    tokens_per_second = completion_tokens / latency_s if latency_s > 0 else 0.0
    text = decode_tokens(tokenizer, output_ids, skip_special_tokens=True)
    logger.info(
        "%s generate prompt_tokens=%d completion_tokens=%d latency_ms=%.1f ttft_ms=%.1f tok_s=%.2f finish=%s",
        engine,
        len(prompt_token_ids),
        completion_tokens,
        latency_s * 1000.0,
        ttft_s * 1000.0,
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
        ttft_s=ttft_s,
        tpot_s=tpot_s,
        engine=engine,
    )
