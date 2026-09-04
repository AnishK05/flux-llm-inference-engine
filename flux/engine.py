"""The Flux inference engine.

A single background worker thread pulls pending requests off a queue, groups
them into a static batch, and decodes the whole batch token-by-token while
reusing a batched KV cache. Generated tokens are streamed back to each caller
through a per-request queue, so the HTTP layer can forward them as
Server-Sent Events or accumulate them into a single response.

The batched decode path uses left-padding plus explicit ``position_ids`` so that
prompts of different lengths remain numerically correct inside one batch.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from flux.config import EngineConfig
from flux.sampling import SamplingParams, sample_token

logger = logging.getLogger("flux.engine")

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class PromptTooLongError(ValueError):
    """Raised when a prompt exceeds the configured token budget."""


@dataclass
class _Sequence:
    """Mutable per-request decoding state, owned by the worker thread."""

    request_id: str
    prompt_ids: List[int]
    params: SamplingParams
    max_new_tokens: int
    stop: List[str]
    out_queue: "queue.Queue"
    generated_ids: List[int] = field(default_factory=list)
    emitted_text: str = ""
    finished: bool = False
    finish_reason: Optional[str] = None
    generator: Optional[torch.Generator] = None


@dataclass
class RequestHandle:
    """Caller-facing handle for a submitted request."""

    request_id: str
    prompt_tokens: int
    out_queue: "queue.Queue"


class InferenceEngine:
    """Loads a causal LM and serves batched, streaming text generation."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._queue: "queue.Queue[_Sequence]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self.tokenizer = None
        self.model = None
        self._torch_dtype = _DTYPES.get(self.config.dtype, torch.float32)

    # ------------------------------------------------------------------ setup
    def load(self) -> None:
        """Load the tokenizer and model, then start the worker thread."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model %s on %s", self.config.model, self.config.device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model, torch_dtype=self._torch_dtype
        )
        model.to(self.config.device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model
        self._ready.set()

        self._worker = threading.Thread(target=self._run_loop, name="flux-engine", daemon=True)
        self._worker.start()
        logger.info("Engine ready (max_batch_size=%d)", self.config.max_batch_size)

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        return self._ready.wait(timeout)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def shutdown(self) -> None:
        self._shutdown.set()
        # Unblock the worker if it is waiting on the queue.
        self._queue.put(None)  # type: ignore[arg-type]
        if self._worker is not None:
            self._worker.join(timeout=5)

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        prompt: str,
        params: SamplingParams,
        max_new_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> RequestHandle:
        if self.tokenizer is None:
            raise RuntimeError("Engine is not loaded")

        prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) > self.config.max_prompt_tokens:
            raise PromptTooLongError(
                f"Prompt has {len(prompt_ids)} tokens; limit is "
                f"{self.config.max_prompt_tokens}"
            )

        capped = max(1, min(int(max_new_tokens), self.config.max_new_tokens_cap))
        generator: Optional[torch.Generator] = None
        if params.seed is not None:
            generator = torch.Generator(device=self.config.device)
            generator.manual_seed(int(params.seed))

        request_id = f"req-{id(prompt_ids):x}-{len(prompt_ids)}"
        seq = _Sequence(
            request_id=request_id,
            prompt_ids=prompt_ids,
            params=params.normalized(),
            max_new_tokens=capped,
            stop=list(stop or []),
            out_queue=queue.Queue(),
            generator=generator,
        )
        self._queue.put(seq)
        return RequestHandle(request_id, len(prompt_ids), seq.out_queue)

    # ------------------------------------------------------------------ worker
    def _collect_batch(self) -> Optional[List[_Sequence]]:
        first = self._queue.get()
        if first is None or self._shutdown.is_set():
            return None
        batch = [first]
        while len(batch) < self.config.max_batch_size:
            try:
                nxt = self._queue.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                break
            batch.append(nxt)
        return batch

    def _run_loop(self) -> None:
        while not self._shutdown.is_set():
            batch = self._collect_batch()
            if batch is None:
                break
            try:
                self._process_batch(batch)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Batch failed: %s", exc)
                for seq in batch:
                    if not seq.finished:
                        self._finish(seq, "error")

    @torch.no_grad()
    def _process_batch(self, batch: List[_Sequence]) -> None:
        device = self.config.device
        pad_id = self.tokenizer.pad_token_id

        max_len = max(len(seq.prompt_ids) for seq in batch)
        input_rows, mask_rows = [], []
        for seq in batch:
            pad = max_len - len(seq.prompt_ids)
            input_rows.append([pad_id] * pad + seq.prompt_ids)
            mask_rows.append([0] * pad + [1] * len(seq.prompt_ids))

        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp(min=0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        # Position of the most recent real token for each sequence.
        cur_positions = attention_mask.sum(-1) - 1

        steps = max(seq.max_new_tokens for seq in batch)
        for _ in range(steps):
            next_tokens = torch.full((len(batch),), pad_id, dtype=torch.long, device=device)
            for i, seq in enumerate(batch):
                if seq.finished:
                    continue
                token = sample_token(
                    logits[i], seq.params, seq.generated_ids, seq.generator
                )
                next_tokens[i] = token
                self._accept_token(seq, token)

            if all(seq.finished for seq in batch):
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones((len(batch), 1), dtype=torch.long, device=device)],
                dim=1,
            )
            cur_positions = cur_positions + 1
            outputs = self.model(
                input_ids=next_tokens.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=cur_positions.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        for seq in batch:
            if not seq.finished:
                self._finish(seq, "length")

    # ----------------------------------------------------------- token handling
    def _accept_token(self, seq: _Sequence, token: int) -> None:
        """Record a sampled token, emit any new text, and check stop rules."""

        if token == self.tokenizer.eos_token_id:
            self._finish(seq, "stop")
            return

        seq.generated_ids.append(token)
        full_text = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)

        # Enforce stop strings against the full decoded text.
        for stop_str in seq.stop:
            if stop_str and stop_str in full_text:
                full_text = full_text.split(stop_str)[0]
                delta = full_text[len(seq.emitted_text):]
                if delta:
                    seq.emitted_text = full_text
                    seq.out_queue.put({"type": "token", "text": delta})
                self._finish(seq, "stop")
                return

        delta = full_text[len(seq.emitted_text):]
        if delta:
            seq.emitted_text = full_text
            seq.out_queue.put({"type": "token", "text": delta})

        if len(seq.generated_ids) >= seq.max_new_tokens:
            self._finish(seq, "length")

    def _finish(self, seq: _Sequence, reason: str) -> None:
        if seq.finished:
            return
        seq.finished = True
        seq.finish_reason = reason
        seq.out_queue.put(
            {
                "type": "done",
                "finish_reason": reason,
                "completion_tokens": len(seq.generated_ids),
                "text": seq.emitted_text,
            }
        )

    # ----------------------------------------------------------------- helpers
    def info(self) -> Dict[str, object]:
        return {
            "model": self.config.model,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "max_batch_size": self.config.max_batch_size,
            "ready": self.is_ready,
        }
