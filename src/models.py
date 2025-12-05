"""
Pydantic models for request/response schemas.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


# OpenAI Models
class OpenAIChatMessage(BaseModel):
    """OpenAI chat message."""

    role: str
    content: Union[str, List[Dict[str, Any]]]
    reasoning_content: Optional[str] = None


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str
    messages: List[OpenAIChatMessage]
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


class OpenAIChatCompletionChoice(BaseModel):
    """OpenAI chat completion choice."""

    index: int
    message: OpenAIChatMessage
    finish_reason: Optional[str] = None


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatCompletionChoice]


class OpenAIDelta(BaseModel):
    """OpenAI streaming delta."""

    content: Optional[str] = None
    reasoning_content: Optional[str] = None


class OpenAIChatCompletionStreamChoice(BaseModel):
    """OpenAI streaming choice."""

    index: int
    delta: OpenAIDelta
    finish_reason: Optional[str] = None


class OpenAIChatCompletionStreamResponse(BaseModel):
    """OpenAI streaming response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatCompletionStreamChoice]


# Gemini Models
class GeminiPart(BaseModel):
    """Gemini content part."""

    text: str


class GeminiContent(BaseModel):
    """Gemini message content."""

    role: str
    parts: List[GeminiPart]


class GeminiRequest(BaseModel):
    """Gemini API request."""

    contents: List[GeminiContent]


class GeminiCandidate(BaseModel):
    """Gemini response candidate."""

    content: GeminiContent
    finish_reason: Optional[str] = None
    index: int


class GeminiResponse(BaseModel):
    """Gemini API response."""

    candidates: List[GeminiCandidate]
