"""Transformers for converting between API formats."""

from .openai import (
    openai_request_to_gemini,
    gemini_response_to_openai,
    gemini_stream_chunk_to_openai,
)
from .anthropic import (
    anthropic_request_to_gemini,
    gemini_response_to_anthropic,
    AnthropicStreamProcessor,
    format_sse_event,
    create_anthropic_error,
)

__all__ = [
    # OpenAI
    "openai_request_to_gemini",
    "gemini_response_to_openai",
    "gemini_stream_chunk_to_openai",
    # Anthropic
    "anthropic_request_to_gemini",
    "gemini_response_to_anthropic",
    "AnthropicStreamProcessor",
    "format_sse_event",
    "create_anthropic_error",
]
