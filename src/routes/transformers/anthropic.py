"""
Anthropic Format Transformers - Handles conversion between Anthropic/Claude and Gemini API formats.
This module contains all the logic for transforming requests and responses between the two formats.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ...schemas import AnthropicMessagesRequest
from ...config import DEFAULT_SAFETY_SETTINGS
from ...models import (
    get_base_model_name,
    should_include_thoughts,
)


def anthropic_request_to_gemini(
    anthropic_request: AnthropicMessagesRequest,
) -> Dict[str, Any]:
    """
    Transform an Anthropic messages request to Gemini format.

    Args:
        anthropic_request: Anthropic format request

    Returns:
        Dictionary in Gemini API format
    """
    contents = []

    # Handle system message/prompt
    system_instruction = None
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            system_instruction = {"parts": [{"text": anthropic_request.system}]}
        elif isinstance(anthropic_request.system, list):
            # System can be a list of content blocks
            system_parts = []
            for block in anthropic_request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_parts.append({"text": block.get("text", "")})
            if system_parts:
                system_instruction = {"parts": system_parts}

    # Process each message in the conversation
    for message in anthropic_request.messages:
        role = message.role

        # Map Anthropic roles to Gemini roles
        if role == "assistant":
            role = "model"
        # "user" stays as "user"

        # Handle different content types
        parts = []
        content = message.content

        if isinstance(content, str):
            # Simple text content
            parts.append({"text": content})
        elif isinstance(content, list):
            # List of content blocks
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "text")

                    if block_type == "text":
                        parts.append({"text": block.get("text", "")})

                    elif block_type == "image":
                        # Handle image content
                        source = block.get("source", {})
                        source_type = source.get("type", "")

                        if source_type == "base64":
                            media_type = source.get("media_type", "image/png")
                            data = source.get("data", "")
                            parts.append(
                                {"inlineData": {"mimeType": media_type, "data": data}}
                            )
                        elif source_type == "url":
                            # URL-based images would need to be fetched
                            # For now, skip or add as text reference
                            url = source.get("url", "")
                            parts.append({"text": f"[Image: {url}]"})

                    elif block_type == "tool_use":
                        # Tool use from assistant - convert to function call
                        parts.append(
                            {
                                "functionCall": {
                                    "name": block.get("name", ""),
                                    "args": block.get("input", {}),
                                }
                            }
                        )

                    elif block_type == "tool_result":
                        # Tool result from user - convert to function response
                        tool_use_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")

                        # Convert content to string if it's a list
                        if isinstance(result_content, list):
                            text_parts = []
                            for item in result_content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    text_parts.append(item.get("text", ""))
                            result_content = "\n".join(text_parts)

                        parts.append(
                            {
                                "functionResponse": {
                                    "name": tool_use_id,  # Use tool_use_id as name
                                    "response": {
                                        "result": result_content,
                                        "is_error": block.get("is_error", False),
                                    },
                                }
                            }
                        )

                    elif block_type == "thinking":
                        # Thinking content - include as thought
                        parts.append(
                            {"text": block.get("thinking", ""), "thought": True}
                        )
                else:
                    # Handle Pydantic models or other objects
                    if hasattr(block, "type"):
                        if block.type == "text":
                            parts.append({"text": getattr(block, "text", "")})

        if parts:
            contents.append({"role": role, "parts": parts})

    # Map Anthropic generation parameters to Gemini format
    generation_config = {}

    if anthropic_request.max_tokens is not None:
        generation_config["maxOutputTokens"] = anthropic_request.max_tokens

    if anthropic_request.temperature is not None:
        generation_config["temperature"] = anthropic_request.temperature

    if anthropic_request.top_p is not None:
        generation_config["topP"] = anthropic_request.top_p

    if anthropic_request.top_k is not None:
        generation_config["topK"] = anthropic_request.top_k

    if anthropic_request.stop_sequences:
        generation_config["stopSequences"] = anthropic_request.stop_sequences

    # Build the request payload
    request_payload = {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": get_base_model_name(anthropic_request.model),
    }

    # Add system instruction if present
    if system_instruction:
        request_payload["systemInstruction"] = system_instruction

    # Handle tools - check for web_search and function tools
    tools_list: List[Dict[str, Any]] = []
    has_web_search = False

    # Handle tools (function declarations and web_search)
    if anthropic_request.tools:
        for tool in anthropic_request.tools:
            # Check for web_search tool (custom extension)
            if hasattr(tool, "type") and tool.type in (
                "web_search",
                "web_search_preview",
            ):
                has_web_search = True
                continue
            elif hasattr(tool, "name") and tool.name in (
                "web_search",
                "web_search_preview",
            ):
                has_web_search = True
                continue

            # Regular function tool
            tool_def = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": tool.input_schema.properties or {},
                    "required": tool.input_schema.required or [],
                },
            }
            tools_list.append({"functionDeclarations": [tool_def]})

    # Add googleSearch tool if web_search was requested
    if has_web_search:
        tools_list.insert(0, {"googleSearch": {}})

    if tools_list:
        request_payload["tools"] = tools_list

    # Add thinking configuration
    if "gemini-2.5-flash-image" not in anthropic_request.model:
        thinking_budget = None

        # Check if thinking config is provided in request
        if anthropic_request.thinking:
            if (
                anthropic_request.thinking.type == "enabled"
                and anthropic_request.thinking.budget_tokens
            ):
                thinking_budget = anthropic_request.thinking.budget_tokens
            elif anthropic_request.thinking.type == "disabled":
                # Use minimal thinking
                base_model = get_base_model_name(anthropic_request.model)
                if "gemini-2.5-flash" in base_model:
                    thinking_budget = 0
                elif "gemini-2.5-pro" in base_model or "gemini-3-pro" in base_model:
                    thinking_budget = 128
        else:
            # Default: auto thinking
            thinking_budget = -1

        if thinking_budget is not None:
            request_payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
                "includeThoughts": should_include_thoughts(anthropic_request.model),
            }

    return request_payload


def gemini_response_to_anthropic(
    gemini_response: Dict[str, Any],
    model: str,
    input_tokens: int = 0,
    include_thinking: bool = False,
) -> Dict[str, Any]:
    """
    Transform a Gemini API response to Anthropic messages format.

    Args:
        gemini_response: Response from Gemini API
        model: Model name to include in response
        input_tokens: Estimated input token count
        include_thinking: Whether to include thinking blocks in response (default False)

    Returns:
        Dictionary in Anthropic messages format
    """
    content_blocks = []
    stop_reason = "end_turn"
    output_tokens = 0

    for candidate in gemini_response.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])

        for part in parts:
            # Handle text parts
            if part.get("text") is not None:
                if part.get("thought", False):
                    # This is thinking content - only include if explicitly requested
                    if include_thinking:
                        content_blocks.append(
                            {"type": "thinking", "thinking": part.get("text", "")}
                        )
                    # Skip thinking blocks if not requested
                else:
                    # Regular text content
                    content_blocks.append(
                        {"type": "text", "text": part.get("text", "")}
                    )

            # Handle function calls (tool use)
            elif part.get("functionCall"):
                func_call = part["functionCall"]
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": func_call.get("name", ""),
                        "input": func_call.get("args", {}),
                    }
                )
                stop_reason = "tool_use"

            # Handle inline data (images in response)
            elif part.get("inlineData"):
                inline = part["inlineData"]
                mime = inline.get("mimeType", "image/png")
                data = inline.get("data", "")
                # Convert to markdown image reference in text
                content_blocks.append(
                    {
                        "type": "text",
                        "text": f"![Generated Image](data:{mime};base64,{data})",
                    }
                )

        # Map finish reason
        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "STOP":
            stop_reason = "end_turn"
        elif finish_reason == "MAX_TOKENS":
            stop_reason = "max_tokens"
        elif finish_reason in ["SAFETY", "RECITATION"]:
            stop_reason = "end_turn"  # Anthropic doesn't have content_filter

    # Get token usage from response metadata if available
    usage_metadata = gemini_response.get("usageMetadata", {})
    input_tokens = usage_metadata.get("promptTokenCount", input_tokens)
    output_tokens = usage_metadata.get(
        "candidatesTokenCount", len(str(content_blocks)) // 4
    )

    # Ensure we have at least one content block
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def create_anthropic_stream_message_start(
    model: str, message_id: str
) -> Dict[str, Any]:
    """
    Create the initial message_start event for streaming.

    Args:
        model: Model name
        message_id: Unique message ID

    Returns:
        message_start event dictionary
    """
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }


def create_anthropic_content_block_start(
    index: int, block_type: str = "text"
) -> Dict[str, Any]:
    """
    Create a content_block_start event.

    Args:
        index: Content block index
        block_type: Type of content block ("text", "thinking", "tool_use")

    Returns:
        content_block_start event dictionary
    """
    if block_type == "text":
        content_block = {"type": "text", "text": ""}
    elif block_type == "thinking":
        content_block = {"type": "thinking", "thinking": ""}
    elif block_type == "tool_use":
        content_block = {
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": "",
            "input": {},
        }
    else:
        content_block = {"type": block_type}

    return {
        "type": "content_block_start",
        "index": index,
        "content_block": content_block,
    }


def create_anthropic_content_block_delta(
    index: int, delta_type: str, content: str
) -> Dict[str, Any]:
    """
    Create a content_block_delta event.

    Args:
        index: Content block index
        delta_type: Type of delta ("text_delta", "thinking_delta", "input_json_delta")
        content: The delta content

    Returns:
        content_block_delta event dictionary
    """
    if delta_type == "text_delta":
        delta = {"type": "text_delta", "text": content}
    elif delta_type == "thinking_delta":
        delta = {"type": "thinking_delta", "thinking": content}
    elif delta_type == "input_json_delta":
        delta = {"type": "input_json_delta", "partial_json": content}
    else:
        delta = {"type": delta_type, "text": content}

    return {"type": "content_block_delta", "index": index, "delta": delta}


def create_anthropic_content_block_stop(index: int) -> Dict[str, Any]:
    """
    Create a content_block_stop event.

    Args:
        index: Content block index

    Returns:
        content_block_stop event dictionary
    """
    return {"type": "content_block_stop", "index": index}


def create_anthropic_message_delta(
    stop_reason: str = "end_turn", output_tokens: int = 0
) -> Dict[str, Any]:
    """
    Create a message_delta event.

    Args:
        stop_reason: Reason for stopping
        output_tokens: Total output tokens

    Returns:
        message_delta event dictionary
    """
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }


def create_anthropic_message_stop() -> Dict[str, Any]:
    """
    Create a message_stop event.

    Returns:
        message_stop event dictionary
    """
    return {"type": "message_stop"}


def create_anthropic_ping() -> Dict[str, Any]:
    """
    Create a ping event for keeping connection alive.

    Returns:
        ping event dictionary
    """
    return {"type": "ping"}


def create_anthropic_error(error_type: str, message: str) -> Dict[str, Any]:
    """
    Create an error event.

    Args:
        error_type: Type of error (e.g., "invalid_request_error", "api_error")
        message: Error message

    Returns:
        error event dictionary
    """
    return {"type": "error", "error": {"type": error_type, "message": message}}


class AnthropicStreamProcessor:
    """
    Processor for converting Gemini streaming chunks to Anthropic SSE format.
    Maintains state across multiple chunks to properly emit events.
    """

    def __init__(self, model: str, include_thinking: bool = False):
        self.model = model
        self.include_thinking = include_thinking
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.current_block_index = -1
        self.current_block_type = None
        self.is_thinking = False
        self.has_started = False
        self.output_tokens = 0
        self.stop_reason = "end_turn"
        self.pending_tool_call = None

    def process_chunk(self, gemini_chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a Gemini streaming chunk and return Anthropic SSE events.

        Args:
            gemini_chunk: A chunk from Gemini streaming response

        Returns:
            List of Anthropic SSE events to emit
        """
        events = []

        # Emit message_start on first chunk
        if not self.has_started:
            events.append(
                create_anthropic_stream_message_start(self.model, self.message_id)
            )
            self.has_started = True

        for candidate in gemini_chunk.get("candidates", []):
            parts = candidate.get("content", {}).get("parts", [])

            for part in parts:
                # Handle text/thinking parts
                if part.get("text") is not None:
                    is_thought = part.get("thought", False)
                    text = part.get("text", "")

                    if not text:
                        continue

                    # Skip thinking blocks if not requested
                    if is_thought and not self.include_thinking:
                        continue

                    # Determine block type
                    block_type = "thinking" if is_thought else "text"
                    delta_type = "thinking_delta" if is_thought else "text_delta"

                    # Check if we need to start a new block
                    if self.current_block_type != block_type:
                        # Close previous block if any
                        if self.current_block_index >= 0:
                            events.append(
                                create_anthropic_content_block_stop(
                                    self.current_block_index
                                )
                            )

                        # Start new block
                        self.current_block_index += 1
                        self.current_block_type = block_type
                        events.append(
                            create_anthropic_content_block_start(
                                self.current_block_index, block_type
                            )
                        )

                    # Emit content delta
                    events.append(
                        create_anthropic_content_block_delta(
                            self.current_block_index, delta_type, text
                        )
                    )

                    self.output_tokens += len(text) // 4  # Rough estimate

                # Handle function calls
                elif part.get("functionCall"):
                    func_call = part["functionCall"]

                    # Close previous block if any
                    if (
                        self.current_block_index >= 0
                        and self.current_block_type != "tool_use"
                    ):
                        events.append(
                            create_anthropic_content_block_stop(
                                self.current_block_index
                            )
                        )

                    # Start tool_use block
                    self.current_block_index += 1
                    self.current_block_type = "tool_use"

                    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    events.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": func_call.get("name", ""),
                                "input": {},
                            },
                        }
                    )

                    # Emit input as delta
                    input_json = json.dumps(func_call.get("args", {}))
                    events.append(
                        create_anthropic_content_block_delta(
                            self.current_block_index, "input_json_delta", input_json
                        )
                    )

                    self.stop_reason = "tool_use"

            # Check finish reason
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "STOP":
                self.stop_reason = "end_turn"
            elif finish_reason == "MAX_TOKENS":
                self.stop_reason = "max_tokens"

        # Update token count from usage metadata
        usage_metadata = gemini_chunk.get("usageMetadata", {})
        if usage_metadata.get("candidatesTokenCount"):
            self.output_tokens = usage_metadata["candidatesTokenCount"]

        return events

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Generate final events to close the stream.

        Returns:
            List of final Anthropic SSE events
        """
        events = []

        # Close last content block if any
        if self.current_block_index >= 0:
            events.append(create_anthropic_content_block_stop(self.current_block_index))

        # Emit message_delta with final stats
        events.append(
            create_anthropic_message_delta(self.stop_reason, self.output_tokens)
        )

        # Emit message_stop
        events.append(create_anthropic_message_stop())

        return events


def format_sse_event(event: Dict[str, Any]) -> str:
    """
    Format an event as an SSE data line.

    Args:
        event: Event dictionary

    Returns:
        Formatted SSE string
    """
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
