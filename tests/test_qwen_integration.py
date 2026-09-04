import os

import pytest

from flux.config import HF_MODEL_ID, Settings
from flux.engine.model_loader import load_causal_lm
from flux.engine.naive_engine import NaiveEngine
from flux.engine.types import SamplingParams
from flux.runtime import apply_thread_caps

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("FLUX_RUN_INTEGRATION") != "1",
        reason="set FLUX_RUN_INTEGRATION=1 to download Qwen and run this",
    ),
]


def test_qwen_naive_max_tokens_4() -> None:
    apply_thread_caps("auto")
    settings = Settings(load_model=True, model=HF_MODEL_ID, device="cpu", dtype="fp32")
    model, tokenizer = load_causal_lm(settings)
    engine = NaiveEngine(model, tokenizer, device="cpu")
    result = engine.generate("Say hi.", SamplingParams(max_tokens=4, temperature=0.0))
    assert result.completion_tokens == 4 or result.finish_reason == "stop"
    assert result.prompt_tokens >= 1
    assert result.latency_s > 0
    assert result.text is not None
