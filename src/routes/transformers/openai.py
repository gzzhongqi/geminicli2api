"""
OpenAI Format Transformers - Handles conversion between OpenAI and Gemini API formats.
"""

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ...schemas import ChatCompletionRequest
from ...models import (
    get_base_model_name,
    should_include_thoughts,
)
from ...config import DEFAULT_SAFETY_SETTINGS

# Regex pattern for markdown images
_MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _parse_data_uri(url: str) -> Optional[Dict[str, Any]]:
    """
    Parse a data URI and return inline data dict if it's an image.

    Args:
        url: Data URI string

    Returns:
        InlineData dict or None if not a valid image data URI
    """
    if not url.startswith("data:"):
        return None

    try:
        header, base64_data = url.split(",", 1)
        mime_type = ""
        if ":" in header:
            mime_type = header.split(":", 1)[1].split(";", 1)[0]

        if mime_type.startswith("image/"):
            return {"inlineData": {"mimeType": mime_type, "data": base64_data}}
    except (ValueError, IndexError):
        pass

    return None


def _extract_images_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract markdown images from text and return parts list.

    Args:
        text: Text potentially containing markdown images

    Returns:
        List of Gemini content parts
    """
    if not text:
        return [{"text": ""}]

    parts: List[Dict[str, Any]] = []
    last_idx = 0

    for match in _MARKDOWN_IMAGE_PATTERN.finditer(text):
        url = match.group(1).strip().strip('"').strip("'")

        # Add text before the image
        if match.start() > last_idx:
            before = text[last_idx : match.start()]
            if before:
                parts.append({"text": before})

        # Try to parse as data URI image
        inline_data = _parse_data_uri(url)
        if inline_data:
            parts.append(inline_data)
        else:
            # Keep non-data URIs as text
            parts.append({"text": text[match.start() : match.end()]})

        last_idx = match.end()

    # Add remaining text
    if last_idx < len(text):
        tail = text[last_idx:]
        if tail:
            parts.append({"text": tail})

    return parts if parts else [{"text": text}]


def _extract_content_and_reasoning(
    parts: List[Dict[str, Any]],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Extract content, reasoning, and function calls from Gemini response parts.

    Args:
        parts: List of Gemini content parts

    Returns:
        Tuple of (content, reasoning_content, tool_calls)
    """
    content_parts: List[str] = []
    reasoning_content = ""
    tool_calls: List[Dict[str, Any]] = []

    for part in parts:
        # Text parts
        if part.get("text") is not None:
            if part.get("thought", False):
                reasoning_content += part.get("text", "")
            else:
                content_parts.append(part.get("text", ""))
            continue

        # Function call parts - Gemini format
        if part.get("functionCall"):
            func_call = part["functionCall"]
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": func_call.get("name", ""),
                        "arguments": json.dumps(func_call.get("args", {})),
                    },
                }
            )
            continue

        # Inline image data -> embed as Markdown data URI
        inline = part.get("inlineData")
        if inline and inline.get("data"):
            mime = inline.get("mimeType") or "image/png"
            if isinstance(mime, str) and mime.startswith("image/"):
                data_b64 = inline.get("data")
                content_parts.append(f"![image](data:{mime};base64,{data_b64})")

    content = "\n\n".join(p for p in content_parts if p)
    return content, reasoning_content, tool_calls


def _map_finish_reason(gemini_reason: Optional[str]) -> Optional[str]:
    """
    Map Gemini finish reasons to OpenAI finish reasons.

    Args:
        gemini_reason: Finish reason from Gemini API

    Returns:
        OpenAI-compatible finish reason or None
    """
    if gemini_reason is None:
        return None

    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    }
    return mapping.get(gemini_reason)


def _process_message_content(content: Any) -> List[Dict[str, Any]]:
    """
    Process message content into Gemini parts format.

    Args:
        content: Message content (string or list)

    Returns:
        List of Gemini content parts
    """
    if content is None:
        return []

    if isinstance(content, str):
        return _extract_images_from_text(content)

    if not isinstance(content, list):
        return [{"text": str(content) if content else ""}]

    parts: List[Dict[str, Any]] = []

    for part in content:
        if part.get("type") == "text":
            text_value = part.get("text", "") or ""
            parts.extend(_extract_images_from_text(text_value))

        elif part.get("type") == "image_url":
            image_url = part.get("image_url", {}).get("url")
            if image_url:
                try:
                    mime_type, base64_part = image_url.split(";")
                    _, mime_type = mime_type.split(":")
                    _, base64_data = base64_part.split(",")
                    parts.append(
                        {"inlineData": {"mimeType": mime_type, "data": base64_data}}
                    )
                except ValueError:
                    continue

    return parts


def _transform_openai_tools_to_gemini(
    tools: List[Any],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Transform OpenAI tools format to Gemini function declarations.

    Args:
        tools: List of OpenAI tool definitions

    Returns:
        Tuple of (function_declarations list, has_web_search bool)
    """
    function_declarations = []
    has_web_search = False

    for tool in tools:
        if hasattr(tool, "type"):
            tool_type = tool.type
            tool_function = getattr(tool, "function", None)
        else:
            tool_type = tool.get("type", "function")
            tool_function = tool.get("function", {})

        # Check for web_search tool
        if tool_type in ("web_search", "web_search_preview"):
            has_web_search = True
            continue

        if tool_type != "function":
            continue

        if not tool_function:
            continue

        # Extract function details
        if hasattr(tool_function, "name"):
            name = tool_function.name
            description = tool_function.description
            parameters = tool_function.parameters
        else:
            name = tool_function.get("name")
            description = tool_function.get("description")
            parameters = tool_function.get("parameters")

        if not name:
            continue

        func_decl: Dict[str, Any] = {"name": name}

        if description:
            func_decl["description"] = description

        if parameters:
            # Gemini expects parameters without $schema and additionalProperties
            params = dict(parameters)
            params.pop("$schema", None)
            params.pop("additionalProperties", None)
            func_decl["parameters"] = params

        function_declarations.append(func_decl)

    return function_declarations, has_web_search


def _transform_tool_choice_to_gemini(
    tool_choice: Any,
) -> Optional[Dict[str, Any]]:
    """
    Transform OpenAI tool_choice to Gemini function calling config.

    Args:
        tool_choice: OpenAI tool_choice value

    Returns:
        Gemini functionCallingConfig or None
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        mode_mapping = {
            "auto": "AUTO",
            "none": "NONE",
            "required": "ANY",
        }
        mode = mode_mapping.get(tool_choice)
        if mode:
            return {"mode": mode}
    elif isinstance(tool_choice, dict):
        # Specific function choice: {"type": "function", "function": {"name": "func_name"}}
        if tool_choice.get("type") == "function":
            func_name = tool_choice.get("function", {}).get("name")
            if func_name:
                return {"mode": "ANY", "allowedFunctionNames": [func_name]}

    return None


def _build_thinking_config(
    model: str, reasoning_effort: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Build thinking configuration for the model.

    Args:
        model: Model name
        reasoning_effort: Optional reasoning effort level (none/off, minimal, low, medium, high, max)

    Returns:
        Thinking config dict or None
    """
    if "gemini-2.5-flash-image" in model:
        return None

    base_model = get_base_model_name(model)

    # Model-specific max budgets
    max_budgets = {
        "gemini-2.5-flash": 24576,
        "gemini-2.5-pro": 32768,
        "gemini-3-pro": 45000,
    }
    # Model-specific minimal budgets (off for flash, minimal for pro)
    minimal_budgets = {
        "gemini-2.5-flash": 0,
        "gemini-2.5-pro": 128,
        "gemini-3-pro": 128,
    }

    thinking_budget: Optional[int] = None

    if reasoning_effort:
        effort_budgets = {
            # Disable thinking completely
            "none": minimal_budgets,
            "off": minimal_budgets,
            "disabled": minimal_budgets,
            # Minimal thinking
            "minimal": minimal_budgets,
            # Low thinking budget
            "low": {"default": 1000},
            # Auto/default thinking
            "medium": {"default": -1},
            # Maximum thinking
            "high": max_budgets,
            "max": max_budgets,
        }

        if reasoning_effort in effort_budgets:
            budgets = effort_budgets[reasoning_effort]
            for key, value in budgets.items():
                if key == "default" or key in base_model:
                    thinking_budget = value
                    break
    else:
        # Default: auto thinking
        thinking_budget = -1

    if thinking_budget is not None:
        return {
            "thinkingBudget": thinking_budget,
            "includeThoughts": should_include_thoughts(model),
        }

    return None


def openai_request_to_gemini(
    openai_request: ChatCompletionRequest,
) -> Dict[str, Any]:
    """
    Transform an OpenAI chat completion request to Gemini format.

    Args:
        openai_request: OpenAI format request

    Returns:
        Dictionary in Gemini API format
    """
    contents: List[Dict[str, Any]] = []

    # Process each message
    for message in openai_request.messages:
        role = message.role

        # Handle tool/function response messages
        if role == "tool":
            # Tool response message - convert to Gemini functionResponse format
            tool_call_id = getattr(message, "tool_call_id", None)
            func_name = getattr(message, "name", None) or "unknown"
            msg_content = message.content if message.content else ""

            # Parse content as JSON if possible, otherwise wrap as string
            try:
                if isinstance(msg_content, str):
                    response_data = json.loads(msg_content) if msg_content else {}
                else:
                    response_data = {"result": str(msg_content)}
            except (json.JSONDecodeError, TypeError):
                response_data = {
                    "result": msg_content
                    if isinstance(msg_content, str)
                    else str(msg_content)
                }

            parts = [
                {
                    "functionResponse": {
                        "name": func_name,
                        "response": response_data,
                    }
                }
            ]
            contents.append({"role": "user", "parts": parts})
            continue

        # Handle assistant messages with tool_calls
        if role == "assistant":
            role = "model"
            tool_calls = getattr(message, "tool_calls", None)

            if tool_calls:
                # Assistant message with function calls
                parts: List[Dict[str, Any]] = []

                # Add text content if present
                if message.content:
                    text_parts = _process_message_content(message.content)
                    parts.extend(text_parts)

                # Add function calls
                for tool_call in tool_calls:
                    if hasattr(tool_call, "function"):
                        func = tool_call.function
                        func_name = (
                            func.get("name")
                            if isinstance(func, dict)
                            else getattr(func, "name", "")
                        )
                        func_args_str = (
                            func.get("arguments")
                            if isinstance(func, dict)
                            else getattr(func, "arguments", "{}")
                        )
                    else:
                        func = tool_call.get("function", {})
                        func_name = func.get("name", "")
                        func_args_str = func.get("arguments", "{}")

                    # Parse arguments JSON string to dict
                    try:
                        func_args = json.loads(func_args_str) if func_args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        func_args = {}

                    parts.append(
                        {
                            "functionCall": {
                                "name": func_name,
                                "args": func_args,
                            }
                        }
                    )

                contents.append({"role": role, "parts": parts})
                continue

        # Standard role mapping
        if role == "assistant":
            role = "model"
        elif role == "system":
            role = "user"

        parts = _process_message_content(message.content)
        if parts:  # Only add if there are parts
            contents.append({"role": role, "parts": parts})

    # Build generation config
    generation_config: Dict[str, Any] = {}

    if openai_request.temperature is not None:
        generation_config["temperature"] = openai_request.temperature
    if openai_request.top_p is not None:
        generation_config["topP"] = openai_request.top_p
    if openai_request.max_tokens is not None:
        generation_config["maxOutputTokens"] = openai_request.max_tokens
    if openai_request.stop is not None:
        stops = (
            [openai_request.stop]
            if isinstance(openai_request.stop, str)
            else openai_request.stop
        )
        generation_config["stopSequences"] = stops
    if openai_request.frequency_penalty is not None:
        generation_config["frequencyPenalty"] = openai_request.frequency_penalty
    if openai_request.presence_penalty is not None:
        generation_config["presencePenalty"] = openai_request.presence_penalty
    if openai_request.n is not None:
        generation_config["candidateCount"] = openai_request.n
    if openai_request.seed is not None:
        generation_config["seed"] = openai_request.seed
    if openai_request.response_format:
        if openai_request.response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"

    # Build request payload
    request_payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": get_base_model_name(openai_request.model),
    }

    # Handle tools - either search or function calling
    tools_list: List[Dict[str, Any]] = []
    has_web_search = False

    # Add function declarations if tools are provided
    if openai_request.tools:
        function_declarations, web_search_in_tools = _transform_openai_tools_to_gemini(
            openai_request.tools
        )
        if web_search_in_tools:
            has_web_search = True
        if function_declarations:
            tools_list.append({"functionDeclarations": function_declarations})

    # Add googleSearch tool if web_search was requested
    if has_web_search:
        tools_list.insert(0, {"googleSearch": {}})

    if tools_list:
        request_payload["tools"] = tools_list

    # Add tool choice config if specified
    if openai_request.tool_choice:
        tool_config = _transform_tool_choice_to_gemini(openai_request.tool_choice)
        if tool_config:
            request_payload["toolConfig"] = {"functionCallingConfig": tool_config}

    # Add thinking config
    reasoning_effort = getattr(openai_request, "reasoning_effort", None)
    thinking_config = _build_thinking_config(openai_request.model, reasoning_effort)
    if thinking_config:
        request_payload["generationConfig"]["thinkingConfig"] = thinking_config

    return request_payload


def gemini_response_to_openai(
    gemini_response: Dict[str, Any], model: str
) -> Dict[str, Any]:
    """
    Transform a Gemini API response to OpenAI chat completion format.

    Args:
        gemini_response: Response from Gemini API
        model: Model name to include in response

    Returns:
        Dictionary in OpenAI chat completion format
    """
    choices: List[Dict[str, Any]] = []

    for candidate in gemini_response.get("candidates", []):
        role = candidate.get("content", {}).get("role", "assistant")
        if role == "model":
            role = "assistant"

        parts = candidate.get("content", {}).get("parts", [])
        content, reasoning_content, tool_calls = _extract_content_and_reasoning(parts)

        message: Dict[str, Any] = {"role": role}

        # Add content (can be None if only tool_calls)
        if content or not tool_calls:
            message["content"] = content if content else None

        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        # Add tool_calls if present
        if tool_calls:
            message["tool_calls"] = tool_calls
            # When there are tool_calls, content should be null per OpenAI spec
            if not content:
                message["content"] = None

        # Determine finish reason
        finish_reason = _map_finish_reason(candidate.get("finishReason"))
        # If there are tool_calls and no explicit finish reason, use "tool_calls"
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        choices.append(
            {
                "index": candidate.get("index", 0),
                "message": message,
                "finish_reason": finish_reason,
            }
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def gemini_stream_chunk_to_openai(
    gemini_chunk: Dict[str, Any], model: str, response_id: str
) -> Dict[str, Any]:
    """
    Transform a Gemini streaming response chunk to OpenAI streaming format.

    Args:
        gemini_chunk: Single chunk from Gemini streaming response
        model: Model name to include in response
        response_id: Consistent ID for this streaming response

    Returns:
        Dictionary in OpenAI streaming format
    """
    choices: List[Dict[str, Any]] = []

    for candidate in gemini_chunk.get("candidates", []):
        role = candidate.get("content", {}).get("role", "assistant")
        if role == "model":
            role = "assistant"

        parts = candidate.get("content", {}).get("parts", [])
        content, reasoning_content, tool_calls = _extract_content_and_reasoning(parts)

        delta: Dict[str, Any] = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content

        # Add tool_calls to delta if present
        if tool_calls:
            delta["tool_calls"] = tool_calls

        # Determine finish reason
        finish_reason = _map_finish_reason(candidate.get("finishReason"))
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"

        choices.append(
            {
                "index": candidate.get("index", 0),
                "delta": delta,
                "finish_reason": finish_reason,
            }
        )

    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
