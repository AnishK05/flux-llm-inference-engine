from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: list[str] | str | None = None
    stream: bool = False

    @field_validator("prompt")
    @classmethod
    def prompt_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("prompt must be a non-empty string")
        return value

    @field_validator("stream")
    @classmethod
    def no_stream_yet(cls, value: bool) -> bool:
        if value:
            raise ValueError("streaming is not available on the naive engine (Phase 7)")
        return value


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: Any = None
    finish_reason: str


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage
