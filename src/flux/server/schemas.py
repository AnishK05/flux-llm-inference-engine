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
            raise ValueError("streaming is not available yet (Phase 7)")
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


class ChatMessage(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_ok(cls, value: str) -> str:
        allowed = {"system", "user", "assistant"}
        if value not in allowed:
            raise ValueError(f"role must be one of {sorted(allowed)}")
        return value

    @field_validator("content")
    @classmethod
    def content_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("content must be a non-empty string")
        return value


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def messages_non_empty(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must be non-empty")
        return value

    @field_validator("stream")
    @classmethod
    def no_stream_yet(cls, value: bool) -> bool:
        if value:
            raise ValueError("streaming is not available yet (Phase 7)")
        return value


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage
