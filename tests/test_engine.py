"""Engine-level tests exercising the real batched decode loop."""

from __future__ import annotations

from flux.sampling import SamplingParams


def _collect(handle):
    """Drain a request handle into (text, finish_reason, completion_tokens)."""

    text, reason, count = "", None, 0
    while True:
        msg = handle.out_queue.get(timeout=120)
        if msg["type"] == "token":
            text += msg["text"]
        elif msg["type"] == "done":
            reason = msg["finish_reason"]
            count = msg["completion_tokens"]
            break
    return text, reason, count


def test_generates_tokens(engine):
    handle = engine.submit(
        "The quick brown fox",
        SamplingParams(temperature=0.0),
        max_new_tokens=8,
    )
    text, reason, count = _collect(handle)
    assert count > 0
    assert reason in {"length", "stop"}
    assert isinstance(text, str)


def test_length_finish_reason(engine):
    handle = engine.submit(
        "Once upon a time",
        SamplingParams(temperature=0.0),
        max_new_tokens=5,
    )
    _, reason, count = _collect(handle)
    assert reason == "length"
    assert count == 5


def test_greedy_is_deterministic(engine):
    prompt = "In the beginning"
    a = _collect(engine.submit(prompt, SamplingParams(temperature=0.0), 10))
    b = _collect(engine.submit(prompt, SamplingParams(temperature=0.0), 10))
    assert a[0] == b[0]


def test_stop_string_truncates(engine):
    # Greedy continuation of a counting prompt reliably contains a space; use it
    # as a stop sequence to prove truncation halts generation.
    handle = engine.submit(
        "1 2 3 4 5 6 7 8 9",
        SamplingParams(temperature=0.0),
        max_new_tokens=20,
        stop=[" "],
    )
    text, reason, _ = _collect(handle)
    assert reason == "stop"
    assert " " not in text


def test_batched_requests_are_independent(engine):
    """Two prompts submitted together share a batch but yield distinct output."""

    h1 = engine.submit("The capital of France is", SamplingParams(temperature=0.0), 6)
    h2 = engine.submit("Water is made of", SamplingParams(temperature=0.0), 6)
    t1, _, c1 = _collect(h1)
    t2, _, c2 = _collect(h2)
    assert c1 > 0 and c2 > 0
    assert t1 != t2
