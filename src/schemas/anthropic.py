"""
Pydantic schemas for Anthropic/Claude API format.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class TextContent(BaseModel):
    """Text content block for Anthropic messages."""

    type: str = "text"
    text: str


class ImageSource(BaseModel):
    """Image source for Anthropic image content."""

    type: str  # "base64" or "url"
    media_type: Optional[str] = None  # e.g., "image/jpeg", "image/png"
    data: Optional[str] = None  # base64 encoded data
    url: Optional[str] = None  # URL for url type


class ImageContent(BaseModel):
    """Image content block for Anthropic messages."""

    type: str = "image"
    source: ImageSource


class ToolUseContent(BaseModel):
    """Tool use content block for Anthropic messages."""

    type: str = "tool_use"
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultContent(BaseModel):
    """Tool result content block for Anthropic messages."""

    type: str = "tool_result"
    tool_use_id: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    is_error: Optional[bool] = False


class ThinkingContent(BaseModel):
    """Thinking content block for Anthropic extended thinking."""

    type: str = "thinking"
    thinking: str


class RedactedThinkingContent(BaseModel):
    """Redacted thinking content block."""

    type: str = "redacted_thinking"
    data: str


# Union type for all Anthropic content blocks
ContentBlock = Union[
    TextContent,
    ImageContent,
    ToolUseContent,
    ToolResultContent,
    ThinkingContent,
    RedactedThinkingContent,
    Dict[str, Any],  # Fallback for unknown content types
]


class Message(BaseModel):
    """A single message in an Anthropic conversation."""

    role: str  # "user" or "assistant"
    content: Union[str, List[ContentBlock]]


class ToolInputSchema(BaseModel):
    """JSON schema for tool input."""

    type: str = "object"
    properties: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None

    class Config:
        extra = "allow"


class Tool(BaseModel):
    """Tool definition for Anthropic API."""

    name: str
    description: Optional[str] = None
    input_schema: ToolInputSchema


class ThinkingConfig(BaseModel):
    """Extended thinking configuration."""

    type: str = "enabled"  # "enabled" or "disabled"
    budget_tokens: Optional[int] = None


class MessagesRequest(BaseModel):
    """
    Request body for Anthropic /v1/messages endpoint.

    Supports:
    - Basic text messages
    - Multi-modal content (images)
    - Tool use
    - Extended thinking
    - Streaming
    """

    model: str
    messages: List[Message]
    max_tokens: int
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None

    class Config:
        extra = "allow"


class Usage(BaseModel):
    """Token usage statistics for Anthropic response."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class ResponseMessage(BaseModel):
    """Response message from Anthropic API."""

    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Dict[str, Any]]
    model: str
    stop_reason: Optional[str] = (
        None  # "end_turn", "max_tokens", "stop_sequence", "tool_use"
    )
    stop_sequence: Optional[str] = None
    usage: Usage


# Streaming event types
class StreamMessageStart(BaseModel):
    """Message start event in streaming."""

    type: str = "message_start"
    message: Dict[str, Any]


class StreamContentBlockStart(BaseModel):
    """Content block start event in streaming."""

    type: str = "content_block_start"
    index: int
    content_block: Dict[str, Any]


class StreamContentBlockDelta(BaseModel):
    """Content block delta event in streaming."""

    type: str = "content_block_delta"
    index: int
    delta: Dict[str, Any]


class StreamContentBlockStop(BaseModel):
    """Content block stop event in streaming."""

    type: str = "content_block_stop"
    index: int


class StreamMessageDelta(BaseModel):
    """Message delta event in streaming."""

    type: str = "message_delta"
    delta: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None


class StreamMessageStop(BaseModel):
    """Message stop event in streaming."""

    type: str = "message_stop"


class StreamPing(BaseModel):
    """Ping event for keeping connection alive."""

    type: str = "ping"


class Error(BaseModel):
    """Error response from Anthropic API."""

    type: str = "error"
    error: Dict[str, Any]  # Contains "type" and "message"
