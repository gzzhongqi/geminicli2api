"""
Pydantic schemas for Anthropic/Claude API format.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TextContent(BaseModel):
    """Text content block for Anthropic messages."""

    type: str = Field(default="text", description="Content type, always 'text'")
    text: str = Field(..., description="The text content")


class ImageSource(BaseModel):
    """Image source for Anthropic image content."""

    type: str = Field(
        ...,
        description="Source type: 'base64' or 'url'",
        examples=["base64", "url"],
    )
    media_type: Optional[str] = Field(
        default=None,
        description="MIME type of the image",
        examples=["image/jpeg", "image/png", "image/gif", "image/webp"],
    )
    data: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (for base64 type)",
    )
    url: Optional[str] = Field(
        default=None,
        description="URL of the image (for url type)",
    )


class ImageContent(BaseModel):
    """Image content block for Anthropic messages."""

    type: str = Field(default="image", description="Content type, always 'image'")
    source: ImageSource = Field(..., description="Image source specification")


class ToolUseContent(BaseModel):
    """Tool use content block for Anthropic messages."""

    type: str = Field(default="tool_use", description="Content type, always 'tool_use'")
    id: str = Field(..., description="Unique identifier for this tool use")
    name: str = Field(..., description="Name of the tool to use")
    input: Dict[str, Any] = Field(..., description="Input parameters for the tool")


class ToolResultContent(BaseModel):
    """Tool result content block for Anthropic messages."""

    type: str = Field(
        default="tool_result", description="Content type, always 'tool_result'"
    )
    tool_use_id: str = Field(..., description="ID of the tool use this result is for")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        default=None,
        description="Result content from the tool",
    )
    is_error: Optional[bool] = Field(
        default=False,
        description="Whether this result represents an error",
    )


class ThinkingContent(BaseModel):
    """Thinking content block for Anthropic extended thinking."""

    type: str = Field(default="thinking", description="Content type, always 'thinking'")
    thinking: str = Field(..., description="The model's thinking/reasoning process")


class RedactedThinkingContent(BaseModel):
    """Redacted thinking content block."""

    type: str = Field(
        default="redacted_thinking",
        description="Content type, always 'redacted_thinking'",
    )
    data: str = Field(..., description="Redacted thinking data")


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

    role: str = Field(
        ...,
        description="Role of the message author",
        examples=["user", "assistant"],
    )
    content: Union[str, List[ContentBlock]] = Field(
        ...,
        description="Message content. Can be a string or array of content blocks",
        examples=["Hello, how can I help you?"],
    )


class ToolInputSchema(BaseModel):
    """JSON schema for tool input."""

    type: str = Field(default="object", description="Schema type, typically 'object'")
    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Property definitions for the tool input",
    )
    required: Optional[List[str]] = Field(
        default=None,
        description="List of required property names",
    )

    class Config:
        extra = "allow"


class Tool(BaseModel):
    """Tool definition for Anthropic API."""

    name: str = Field(
        ...,
        description="Unique name for the tool",
        examples=["get_weather", "search_web"],
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of what the tool does",
    )
    input_schema: ToolInputSchema = Field(
        ...,
        description="JSON schema defining the tool's input parameters",
    )


class ThinkingConfig(BaseModel):
    """Extended thinking configuration."""

    type: str = Field(
        default="enabled",
        description="Thinking mode: 'enabled' or 'disabled'",
        examples=["enabled", "disabled"],
    )
    budget_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens for thinking (when enabled)",
        ge=1,
        examples=[1024, 4096, 10000],
    )


class MessagesRequest(BaseModel):
    """
    Request body for Anthropic-compatible /v1/messages endpoint.

    Maps to Gemini models automatically. All Claude model names are supported
    and mapped to appropriate Gemini equivalents.

    Supports:
    - Basic text messages
    - Multi-modal content (images)
    - Tool use
    - Extended thinking
    - Streaming
    """

    model: str = Field(
        ...,
        description="Model ID. Claude models are mapped to Gemini (e.g., claude-3-5-sonnet → gemini-2.5-flash)",
        examples=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
    )
    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )
    max_tokens: int = Field(
        ...,
        description="Maximum number of tokens to generate",
        ge=1,
        examples=[1024, 4096],
    )
    system: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        default=None,
        description="System prompt. Can be a string or array of content blocks",
        examples=["You are a helpful assistant."],
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for the request",
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Custom stop sequences",
        examples=[["END", "STOP"]],
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response using SSE",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0-1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.7, 1.0],
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling probability",
        ge=0.0,
        le=1.0,
        examples=[0.9],
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k sampling parameter",
        ge=1,
    )
    tools: Optional[List[Tool]] = Field(
        default=None,
        description="List of tools available to the model",
    )
    tool_choice: Optional[Dict[str, Any]] = Field(
        default=None,
        description="How the model should use tools",
        examples=[{"type": "auto"}, {"type": "tool", "name": "get_weather"}],
    )
    thinking: Optional[ThinkingConfig] = Field(
        default=None,
        description="Extended thinking configuration",
    )

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello, Claude!"}],
                "temperature": 0.7,
            }
        }


class Usage(BaseModel):
    """Token usage statistics for Anthropic response."""

    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")
    cache_creation_input_tokens: Optional[int] = Field(
        default=None,
        description="Tokens used for cache creation",
    )
    cache_read_input_tokens: Optional[int] = Field(
        default=None,
        description="Tokens read from cache",
    )


class ResponseMessage(BaseModel):
    """Response message from Anthropic-compatible API."""

    id: str = Field(..., description="Unique message identifier")
    type: str = Field(default="message", description="Object type, always 'message'")
    role: str = Field(default="assistant", description="Role, always 'assistant'")
    content: List[Dict[str, Any]] = Field(..., description="Response content blocks")
    model: str = Field(..., description="Model used for generation")
    stop_reason: Optional[str] = Field(
        default=None,
        description="Reason generation stopped",
        examples=["end_turn", "max_tokens", "stop_sequence", "tool_use"],
    )
    stop_sequence: Optional[str] = Field(
        default=None,
        description="The stop sequence that caused generation to stop",
    )
    usage: Usage = Field(..., description="Token usage statistics")


# Streaming event types
class StreamMessageStart(BaseModel):
    """Message start event in streaming."""

    type: str = Field(default="message_start", description="Event type")
    message: Dict[str, Any] = Field(..., description="Initial message data")


class StreamContentBlockStart(BaseModel):
    """Content block start event in streaming."""

    type: str = Field(default="content_block_start", description="Event type")
    index: int = Field(..., description="Index of the content block")
    content_block: Dict[str, Any] = Field(..., description="Initial content block data")


class StreamContentBlockDelta(BaseModel):
    """Content block delta event in streaming."""

    type: str = Field(default="content_block_delta", description="Event type")
    index: int = Field(..., description="Index of the content block")
    delta: Dict[str, Any] = Field(..., description="Delta content")


class StreamContentBlockStop(BaseModel):
    """Content block stop event in streaming."""

    type: str = Field(default="content_block_stop", description="Event type")
    index: int = Field(..., description="Index of the content block that stopped")


class StreamMessageDelta(BaseModel):
    """Message delta event in streaming."""

    type: str = Field(default="message_delta", description="Event type")
    delta: Dict[str, Any] = Field(..., description="Message delta data")
    usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Updated usage statistics"
    )


class StreamMessageStop(BaseModel):
    """Message stop event in streaming."""

    type: str = Field(default="message_stop", description="Event type")


class StreamPing(BaseModel):
    """Ping event for keeping connection alive."""

    type: str = Field(default="ping", description="Event type")


class Error(BaseModel):
    """Error response from Anthropic-compatible API."""

    type: str = Field(default="error", description="Response type, always 'error'")
    error: Dict[str, Any] = Field(
        ...,
        description="Error details containing 'type' and 'message'",
    )
