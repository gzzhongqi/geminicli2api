"""
OpenAI Format Transformers - Handles conversion between OpenAI and Gemini API formats.
This module contains all the logic for transforming requests and responses between the two formats.
"""
import json
import time
import uuid
import re
from typing import Dict, Any

from .models import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from .config import (
    DEFAULT_SAFETY_SETTINGS,
    is_search_model,
    get_base_model_name,
    get_thinking_budget,
    should_include_thoughts,
    is_nothinking_model,
    is_maxthinking_model
)


def openai_request_to_gemini(openai_request: OpenAIChatCompletionRequest) -> Dict[str, Any]:
    """
    Transform an OpenAI chat completion request to Gemini format.
    
    Args:
        openai_request: OpenAI format request
        
    Returns:
        Dictionary in Gemini API format
    """
    contents = []
    
    # Process each message in the conversation
    for message in openai_request.messages:
        role = message.role
        
        # Map OpenAI roles to Gemini roles
        if role == "assistant":
            role = "model"
        elif role == "system":
            role = "user"  # Gemini treats system messages as user messages
        
        # Handle different content types (string vs list of parts)
        if isinstance(message.content, list):
            parts = []
            for part in message.content:
                if part.get("type") == "text":
                    text_value = part.get("text", "") or ""
                    # Extract Markdown images (data URIs) into inline image parts, preserving surrounding text
                    pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
                    matches = list(pattern.finditer(text_value))
                    if not matches:
                        parts.append({"text": text_value})
                    else:
                        last_idx = 0
                        for m in matches:
                            url = m.group(1).strip().strip('"').strip("'")
                            # Emit text before the image
                            if m.start() > last_idx:
                                before = text_value[last_idx:m.start()]
                                if before:
                                    parts.append({"text": before})
                            # Handle data URI images: data:image/png;base64,xxxx
                            if url.startswith("data:"):
                                try:
                                    header, base64_data = url.split(",", 1)
                                    # header looks like: data:image/png;base64
                                    mime_type = ""
                                    if ":" in header:
                                        mime_type = header.split(":", 1)[1].split(";", 1)[0] or ""
                                    # Only convert to inlineData if it's an image
                                    if mime_type.startswith("image/"):
                                        parts.append({
                                            "inlineData": {
                                                "mimeType": mime_type,
                                                "data": base64_data
                                            }
                                        })
                                    else:
                                        # Non-image data URIs: keep as markdown text
                                        parts.append({"text": text_value[m.start():m.end()]})
                                except Exception:
                                    # Fallback: keep original markdown as text if parsing fails
                                    parts.append({"text": text_value[m.start():m.end()]})
                            else:
                                # Non-data URIs: keep markdown as text (cannot inline without fetching)
                                parts.append({"text": text_value[m.start():m.end()]})
                            last_idx = m.end()
                        # Tail text after last image
                        if last_idx < len(text_value):
                            tail = text_value[last_idx:]
                            if tail:
                                parts.append({"text": tail})
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url")
                    if image_url:
                        # Parse data URI: "data:image/jpeg;base64,{base64_image}"
                        try:
                            mime_type, base64_data = image_url.split(";")
                            _, mime_type = mime_type.split(":")
                            _, base64_data = base64_data.split(",")
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": base64_data
                                }
                            })
                        except ValueError:
                            continue
            contents.append({"role": role, "parts": parts})
        else:
            # Simple text content; extract Markdown images (data URIs) into inline image parts
            text = message.content or ""
            parts = []
            # Convert Markdown images: ![alt](data:<mimeType>;base64,<data>)
            pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
            last_idx = 0
            for m in pattern.finditer(text):
                url = m.group(1).strip().strip('"').strip("'")
                # Emit text before the image
                if m.start() > last_idx:
                    before = text[last_idx:m.start()]
                    if before:
                        parts.append({"text": before})
                # Handle data URI images: data:image/png;base64,xxxx
                if url.startswith("data:"):
                    try:
                        header, base64_data = url.split(",", 1)
                        # header looks like: data:image/png;base64
                        mime_type = ""
                        if ":" in header:
                            mime_type = header.split(":", 1)[1].split(";", 1)[0] or ""
                        # Only convert to inlineData if it's an image
                        if mime_type.startswith("image/"):
                            parts.append({
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": base64_data
                                }
                            })
                        else:
                            # Non-image data URIs: keep as markdown text
                            parts.append({"text": text[m.start():m.end()]})
                    except Exception:
                        # Fallback: keep original markdown as text if parsing fails
                        parts.append({"text": text[m.start():m.end()]})
                else:
                    # Non-data URIs: keep markdown as text (cannot inline without fetching)
                    parts.append({"text": text[m.start():m.end()]})
                last_idx = m.end()
            # Tail text after last image
            if last_idx < len(text):
                tail = text[last_idx:]
                if tail:
                    parts.append({"text": tail})
            contents.append({"role": role, "parts": parts if parts else [{"text": text}]})
    
    # Map OpenAI generation parameters to Gemini format
    generation_config = {}
    if openai_request.temperature is not None:
        generation_config["temperature"] = openai_request.temperature
    if openai_request.top_p is not None:
        generation_config["topP"] = openai_request.top_p
    if openai_request.max_tokens is not None:
        generation_config["maxOutputTokens"] = openai_request.max_tokens
    if openai_request.stop is not None:
        # Gemini supports stop sequences
        if isinstance(openai_request.stop, str):
            generation_config["stopSequences"] = [openai_request.stop]
        elif isinstance(openai_request.stop, list):
            generation_config["stopSequences"] = openai_request.stop
    if openai_request.frequency_penalty is not None:
        # Map frequency_penalty to Gemini's frequencyPenalty
        generation_config["frequencyPenalty"] = openai_request.frequency_penalty
    if openai_request.presence_penalty is not None:
        # Map presence_penalty to Gemini's presencePenalty
        generation_config["presencePenalty"] = openai_request.presence_penalty
    if openai_request.n is not None:
        # Map n (number of completions) to Gemini's candidateCount
        generation_config["candidateCount"] = openai_request.n
    if openai_request.seed is not None:
        # Gemini supports seed for reproducible outputs
        generation_config["seed"] = openai_request.seed
    if openai_request.response_format is not None:
        # Handle JSON mode if specified
        if openai_request.response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"
    
    # generation_config["enableEnhancedCivicAnswers"] = False

    # Build the request payload
    request_payload = {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": get_base_model_name(openai_request.model)  # Use base model name for API call
    }
    
    # Add Google Search grounding for search models
    if is_search_model(openai_request.model):
        request_payload["tools"] = [{"googleSearch": {}}]
    
    if "gemini-2.5-flash-image" not in openai_request.model:
        # Add thinking configuration for thinking models
        thinking_budget = None
        
        # Check if model is an explicit thinking variant (nothinking or maxthinking)
        if is_nothinking_model(openai_request.model) or is_maxthinking_model(openai_request.model):
            # For explicit thinking variants, ignore reasoning_effort and use variant-specific budget
            thinking_budget = get_thinking_budget(openai_request.model)
        else:
            # For regular models, check if reasoning_effort was provided in the request
            reasoning_effort = getattr(openai_request, 'reasoning_effort', None)
            if reasoning_effort:
                base_model = get_base_model_name(openai_request.model)
                if reasoning_effort == "minimal":
                    # Use same budget as nothinking variants
                    if "gemini-2.5-flash" in base_model:
                        thinking_budget = 0
                    elif "gemini-2.5-pro" in base_model or "gemini-3-pro" in base_model:
                        thinking_budget = 128
                elif reasoning_effort == "low":
                    thinking_budget = 1000
                elif reasoning_effort == "medium":
                    thinking_budget = -1
                elif reasoning_effort == "high":
                    # Use same budget as maxthinking variants
                    if "gemini-2.5-flash" in base_model:
                        thinking_budget = 24576
                    elif "gemini-2.5-pro" in base_model:
                        thinking_budget = 32768
                    elif "gemini-3-pro" in base_model:
                        thinking_budget = 45000
            else:
                # No reasoning_effort provided, use default thinking budget
                thinking_budget = get_thinking_budget(openai_request.model)
        
        if thinking_budget is not None:
            request_payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
                "includeThoughts": should_include_thoughts(openai_request.model)
            }
    
    return request_payload


def gemini_response_to_openai(gemini_response: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Transform a Gemini API response to OpenAI chat completion format.
    
    Args:
        gemini_response: Response from Gemini API
        model: Model name to include in response
        
    Returns:
        Dictionary in OpenAI chat completion format
    """
    choices = []
    
    for candidate in gemini_response.get("candidates", []):
        role = candidate.get("content", {}).get("role", "assistant")
        
        # Map Gemini roles back to OpenAI roles
        if role == "model":
            role = "assistant"
        
        # Extract and separate thinking tokens from regular content
        parts = candidate.get("content", {}).get("parts", [])
        content_parts = []
        reasoning_content = ""
        
        for part in parts:
            # Text parts (may include thinking tokens)
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
                continue

        content = "\n\n".join([p for p in content_parts if p is not None])
        
        # Build message object
        message = {
            "role": role,
            "content": content,
        }
        
        # Add reasoning_content if there are thinking tokens
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        
        choices.append({
            "index": candidate.get("index", 0),
            "message": message,
            "finish_reason": _map_finish_reason(candidate.get("finishReason")),
        })
    
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def gemini_stream_chunk_to_openai(gemini_chunk: Dict[str, Any], model: str, response_id: str) -> Dict[str, Any]:
    """
    Transform a Gemini streaming response chunk to OpenAI streaming format.
    
    Args:
        gemini_chunk: Single chunk from Gemini streaming response
        model: Model name to include in response
        response_id: Consistent ID for this streaming response
        
    Returns:
        Dictionary in OpenAI streaming format
    """
    choices = []
    
    for candidate in gemini_chunk.get("candidates", []):
        role = candidate.get("content", {}).get("role", "assistant")
        
        # Map Gemini roles back to OpenAI roles
        if role == "model":
            role = "assistant"
        
        # Extract and separate thinking tokens from regular content
        parts = candidate.get("content", {}).get("parts", [])
        content_parts = []
        reasoning_content = ""
        
        for part in parts:
            # Text parts (may include thinking tokens)
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
                continue

        content = "\n\n".join([p for p in content_parts if p is not None])
        
        # Build delta object
        delta = {}
        if content:
            delta["content"] = content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        
        choices.append({
            "index": candidate.get("index", 0),
            "delta": delta,
            "finish_reason": _map_finish_reason(candidate.get("finishReason")),
        })
    
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def _map_finish_reason(gemini_reason: str) -> str:
    """
    Map Gemini finish reasons to OpenAI finish reasons.
    
    Args:
        gemini_reason: Finish reason from Gemini API
        
    Returns:
        OpenAI-compatible finish reason
    """
    if gemini_reason == "STOP":
        return "stop"
    elif gemini_reason == "MAX_TOKENS":
        return "length"
    elif gemini_reason in ["SAFETY", "RECITATION"]:
        return "content_filter"
    else:
        return None