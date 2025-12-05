"""
Pydantic schemas for OpenAI API format.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message in a conversation."""

    role: str = Field(
        ...,
        description="The role of the message author",
        examples=["user", "assistant", "system"],
    )
    content: Union[str, List[Dict[str, Any]]] = Field(
        ...,
        description="The content of the message. Can be a string or array of content parts for multi-modal input.",
        examples=["Hello, how can I help you today?"],
    )
    reasoning_content: Optional[str] = Field(
        default=None,
        description="Extended thinking/reasoning content (for thinking models)",
    )


class ChatCompletionRequest(BaseModel):
    """
    Request body for OpenAI-compatible chat completions.

    Maps to Gemini models automatically based on the requested model name.
    """

    model: str = Field(
        ...,
        description="Model ID to use. Mapped to Gemini models (e.g., gpt-4o → gemini-2.0-flash)",
        examples=["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
    )
    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response using SSE",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0-2.0). Higher = more random",
        ge=0.0,
        le=2.0,
        examples=[0.7, 1.0],
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling probability. Alternative to temperature",
        ge=0.0,
        le=1.0,
        examples=[0.9, 0.95],
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate",
        ge=1,
        examples=[1024, 4096],
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Stop sequences. Generation stops when these are encountered",
        examples=["END", ["STOP", "END"]],
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description="Frequency penalty (-2.0 to 2.0). Reduces repetition",
        ge=-2.0,
        le=2.0,
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description="Presence penalty (-2.0 to 2.0). Encourages new topics",
        ge=-2.0,
        le=2.0,
    )
    n: Optional[int] = Field(
        default=None,
        description="Number of completions to generate",
        ge=1,
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for deterministic generation",
    )
    response_format: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response format specification (e.g., JSON mode)",
        examples=[{"type": "json_object"}],
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level for thinking models",
        examples=["low", "medium", "high"],
    )

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False,
            }
        }


class ChatCompletionChoice(BaseModel):
    """A single completion choice in the response."""

    index: int = Field(..., description="Index of this choice")
    message: ChatMessage = Field(..., description="The generated message")
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason generation stopped",
        examples=["stop", "length", "content_filter"],
    )


class ChatCompletionResponse(BaseModel):
    """Response from chat completions endpoint."""

    id: str = Field(..., description="Unique identifier for this completion")
    object: str = Field(
        default="chat.completion",
        description="Object type, always 'chat.completion'",
    )
    created: int = Field(
        ..., description="Unix timestamp of when the completion was created"
    )
    model: str = Field(..., description="Model used for completion")
    choices: List[ChatCompletionChoice] = Field(
        ..., description="List of completion choices"
    )


class StreamDelta(BaseModel):
    """Delta content in streaming response."""

    content: Optional[str] = Field(default=None, description="Text content delta")
    reasoning_content: Optional[str] = Field(
        default=None,
        description="Reasoning content delta (for thinking models)",
    )


class StreamChoice(BaseModel):
    """A single choice in streaming response."""

    index: int = Field(..., description="Index of this choice")
    delta: StreamDelta = Field(..., description="Delta content for this chunk")
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason generation stopped (only in final chunk)",
    )


class StreamResponse(BaseModel):
    """Streaming response chunk."""

    id: str = Field(..., description="Unique identifier for this completion")
    object: str = Field(
        default="chat.completion.chunk",
        description="Object type, always 'chat.completion.chunk'",
    )
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[StreamChoice] = Field(..., description="List of streaming choices")
