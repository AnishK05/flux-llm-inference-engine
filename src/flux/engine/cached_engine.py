"""Phase 2 engine: prefill once, then decode with a KV cache.

First token is sampled from prefill logits. Later tokens call `decode` with
`input_ids` of shape `[batch, 1]` and the growing cache.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import torch

from flux.engine.batching import attention_mask_for_decode, extract_row_cache, stack_caches, wrap_past
from flux.engine.kv_utils import cache_seq_len
from flux.engine.naive_engine import pack_result
from flux.engine.sampler import sample_logits
from flux.engine.tokenizer import combined_stop_ids, decode_tokens, encode_chat, encode_text
from flux.engine.types import GenerateResult, SamplingParams

logger = logging.getLogger(__name__)


class CachedEngine:
    def __init__(self, model: Any, tokenizer: Any, device: str = "cpu") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_loaded = True
        self.engine_name = "cached"

    def prefill(self, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, Any]:
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.device)
        cache_position = torch.arange(prompt_ids.shape[1], device=self.device)
        with torch.inference_mode():
            outputs = self.model(
                prompt_ids,
                use_cache=True,
                cache_position=cache_position,
            )
        logits = outputs.logits[:, -1, :]
        return logits, outputs.past_key_values

    def decode(self, last_token: torch.Tensor, kv_cache: Any) -> tuple[torch.Tensor, Any]:
        token_ids = last_token.view(-1, 1).to(self.device)
        past_len = cache_seq_len(kv_cache)
        cache_position = torch.arange(past_len, past_len + token_ids.shape[1], device=self.device)
        with torch.inference_mode():
            outputs = self.model(
                token_ids,
                use_cache=True,
                past_key_values=kv_cache,
                cache_position=cache_position,
            )
        logits = outputs.logits[:, -1, :]
        return logits, outputs.past_key_values

    def decode_batch(
        self, last_tokens: torch.Tensor, caches: list[Any]
    ) -> tuple[torch.Tensor, list[Any]]:
        """Decode one token for B sequences with right-padded stacked KV caches."""
        if not caches:
            raise ValueError("decode_batch requires caches")
        token_ids = last_tokens.view(-1, 1).to(self.device)
        stacked, lengths, max_past = stack_caches(caches)
        past = wrap_past(stacked, huggingface=self._huggingface)
        attn = attention_mask_for_decode(lengths, max_past, device=token_ids.device)
        position_ids = torch.tensor(lengths, dtype=torch.long, device=self.device).view(-1, 1)
        cache_position = torch.tensor([max_past], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            outputs = self.model(
                token_ids,
                use_cache=True,
                past_key_values=past,
                attention_mask=attn,
                position_ids=position_ids,
                cache_position=cache_position,
            )
        logits = outputs.logits[:, -1, :]
        # Keep per-row caches as legacy (K, V) lists so the next stack is uniform.
        new_caches = [
            extract_row_cache(outputs.past_key_values, row, old_len)
            for row, old_len in enumerate(lengths)
        ]
        return logits, new_caches

    @property
    def _huggingface(self) -> bool:
        return hasattr(self.model, "config")

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
        logger.info("cached chat template: %s", templated[:500])
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
        last_token: torch.Tensor | None = None

        with torch.inference_mode():
            logits, kv_cache = self.prefill(input_ids)
            next_id = sample_logits(logits, sampling)
            token = int(next_id.item())
            output_ids.append(token)
            first_token_at = time.perf_counter()
            if token in stop_ids:
                finish_reason = "stop"
            else:
                last_token = next_id
                for _ in range(sampling.max_tokens - 1):
                    logits, kv_cache = self.decode(last_token, kv_cache)
                    next_id = sample_logits(logits, sampling)
                    token = int(next_id.item())
                    output_ids.append(token)
                    if token in stop_ids:
                        finish_reason = "stop"
                        break
                    last_token = next_id

        return pack_result(
            tokenizer=self.tokenizer,
            prompt_token_ids=prompt_token_ids,
            output_ids=output_ids,
            finish_reason=finish_reason,
            started=started,
            first_token_at=first_token_at,
            engine=self.engine_name,
        )
