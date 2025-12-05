"""
OpenAI Format Transformers - Handles conversion between OpenAI and Gemini API formats.
"""

import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import ChatCompletionRequest
from ..models import (
    is_search_model,
    get_base_model_name,
    get_thinking_budget,
    should_include_thoughts,
    is_nothinking_model,
    is_maxthinking_model,
)
from ..config import DEFAULT_SAFETY_SETTINGS

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


def _extract_content_and_reasoning(parts: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Extract content and reasoning from Gemini response parts.

    Args:
        parts: List of Gemini content parts

    Returns:
        Tuple of (content, reasoning_content)
    """
    content_parts: List[str] = []
    reasoning_content = ""

    for part in parts:
        # Text parts
        if part.get("text") is not None:
            if part.get("thought", False):
                reasoning_content += part.get("text", "")
            else:
                content_parts.append(part.get("text", ""))
            continue

        # Inline image data -> embed as Markdown data URI
        inline = part.get("inlineData")
        if inline and inline.get("data"):
            mime = inline.get("mimeType") or "image/png"
            if isinstance(mime, str) and mime.startswith("image/"):
                data_b64 = inline.get("data")
                content_parts.append(f"![image](data:{mime};base64,{data_b64})")

    content = "\n\n".join(p for p in content_parts if p)
    return content, reasoning_content


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


def _build_thinking_config(
    model: str, reasoning_effort: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Build thinking configuration for the model.

    Args:
        model: Model name
        reasoning_effort: Optional reasoning effort level

    Returns:
        Thinking config dict or None
    """
    if "gemini-2.5-flash-image" in model:
        return None

    thinking_budget: Optional[int] = None

    # Explicit thinking variants ignore reasoning_effort
    if is_nothinking_model(model) or is_maxthinking_model(model):
        thinking_budget = get_thinking_budget(model)
    elif reasoning_effort:
        base_model = get_base_model_name(model)
        effort_budgets = {
            "minimal": {
                "gemini-2.5-flash": 0,
                "gemini-2.5-pro": 128,
                "gemini-3-pro": 128,
            },
            "low": {"default": 1000},
            "medium": {"default": -1},
            "high": {
                "gemini-2.5-flash": 24576,
                "gemini-2.5-pro": 32768,
                "gemini-3-pro": 45000,
            },
        }

        if reasoning_effort in effort_budgets:
            budgets = effort_budgets[reasoning_effort]
            for key, value in budgets.items():
                if key == "default" or key in base_model:
                    thinking_budget = value
                    break
    else:
        thinking_budget = get_thinking_budget(model)

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
        if role == "assistant":
            role = "model"
        elif role == "system":
            role = "user"

        parts = _process_message_content(message.content)
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

    # Add search tool if needed
    if is_search_model(openai_request.model):
        request_payload["tools"] = [{"googleSearch": {}}]

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
        content, reasoning_content = _extract_content_and_reasoning(parts)

        message: Dict[str, Any] = {"role": role, "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        choices.append(
            {
                "index": candidate.get("index", 0),
                "message": message,
                "finish_reason": _map_finish_reason(candidate.get("finishReason")),
            }
        )

    return {
        "id": str(uuid.uuid4()),
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
        content, reasoning_content = _extract_content_and_reasoning(parts)

        delta: Dict[str, str] = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content

        choices.append(
            {
                "index": candidate.get("index", 0),
                "delta": delta,
                "finish_reason": _map_finish_reason(candidate.get("finishReason")),
            }
        )

    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }
