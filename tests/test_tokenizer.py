from flux.engine.tokenizer import encode_chat, encode_text, stop_token_ids
from flux.engine.fake_lm import FakeTokenizer


def test_encode_text_rejects_empty() -> None:
    try:
        encode_text(FakeTokenizer(), "")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_chat_template_includes_roles_and_generation_prompt() -> None:
    tokenizer = FakeTokenizer()
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<|system|>" in text
    assert "<|user|>" in text
    assert "Be brief." in text
    assert text.endswith("<|assistant|>")
    ids = encode_chat(
        tokenizer,
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert ids.shape[0] == 1
    assert ids.shape[1] > 1


def test_stop_token_ids_include_eos() -> None:
    tokenizer = FakeTokenizer()
    assert tokenizer.eos_token_id in stop_token_ids(tokenizer)
