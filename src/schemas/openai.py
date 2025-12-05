"""
Pydantic schemas for OpenAI API format.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: Union[str, List[Dict[str, Any]]]
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None

    class Config:
        extra = "allow"


class ChatCompletionChoice(BaseModel):
    """Chat completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class StreamDelta(BaseModel):
    """Streaming delta."""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class StreamChoice(BaseModel):
    """Streaming choice."""

    index: int
    delta: StreamDelta
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    """Streaming response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]
