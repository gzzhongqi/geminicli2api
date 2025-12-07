"""
Responses API Format Transformers - Handles conversion between OpenAI Responses API and Gemini formats.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ...schemas.responses import ResponsesRequest
from ...models import (
    get_base_model_name,
    should_include_thoughts,
)
from ...config import DEFAULT_SAFETY_SETTINGS


def _transform_responses_tools_to_gemini(
    tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Transform Responses API tools to Gemini function declarations.

    Args:
        tools: List of Responses API tool definitions

    Returns:
        Tuple of (function_declarations list, has_web_search bool)
    """
    function_declarations = []
    has_web_search = False

    for tool in tools:
        tool_type = tool.get("type", "function")

        # Handle web search tool
        if tool_type in ("web_search", "web_search_preview"):
            has_web_search = True
            continue

        # Handle function tools
        if tool_type == "function":
            name = tool.get("name")
            if not name:
                # Check nested function structure (OpenAI format)
                func = tool.get("function", {})
                name = func.get("name")
                description = func.get("description")
                parameters = func.get("parameters")
            else:
                # Responses API flat structure
                description = tool.get("description")
                parameters = tool.get("parameters")

            if not name:
                continue

            func_decl: Dict[str, Any] = {"name": name}
            if description:
                func_decl["description"] = description
            if parameters:
                # Clean up parameters for Gemini
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
    Transform tool_choice to Gemini function calling config.

    Args:
        tool_choice: Responses API tool_choice value

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
        if tool_choice.get("type") == "function":
            func_name = tool_choice.get("function", {}).get("name")
            if func_name:
                return {"mode": "ANY", "allowedFunctionNames": [func_name]}

    return None


def _build_thinking_config(
    model: str, reasoning: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Build thinking configuration for the model.

    Args:
        model: Model name
        reasoning: Optional reasoning configuration from request

    Returns:
        Thinking config dict or None
    """
    if "gemini-2.5-flash-image" in model:
        return None

    base_model = get_base_model_name(model)
    thinking_budget: Optional[int] = None

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

    if reasoning:
        # Map Responses API reasoning effort to budget
        effort = reasoning.get("effort", "medium")
        effort_budgets = {
            # Disable thinking
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
        if effort in effort_budgets:
            budgets = effort_budgets[effort]
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


def _process_input_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single input item to Gemini content format.

    Args:
        item: Input item (message or function output)

    Returns:
        Gemini content dict or None
    """
    item_type = item.get("type", "message")

    # Handle function call output
    if item_type == "function_call_output":
        call_id = item.get("call_id", "")
        output = item.get("output", "")

        # Parse output as JSON if possible
        try:
            if isinstance(output, str):
                response_data = json.loads(output) if output else {}
            else:
                response_data = {"result": str(output)}
        except (json.JSONDecodeError, TypeError):
            response_data = {
                "result": output if isinstance(output, str) else str(output)
            }

        # We need to find the function name from context, but for now use a placeholder
        # The actual name should come from matching the call_id with previous function calls
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": item.get("name", "function"),
                        "response": response_data,
                    }
                }
            ],
        }

    # Handle regular messages
    role = item.get("role", "user")
    content = item.get("content", "")

    # Map roles
    gemini_role = "model" if role == "assistant" else "user"

    parts: List[Dict[str, Any]] = []

    # Handle content
    if isinstance(content, str):
        if content:
            parts.append({"text": content})
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append({"text": part.get("text", "")})
                elif part.get("type") == "input_text":
                    parts.append({"text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        try:
                            header, base64_data = image_url.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            parts.append(
                                {
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": base64_data,
                                    }
                                }
                            )
                        except (ValueError, IndexError):
                            pass
            elif isinstance(part, str):
                parts.append({"text": part})

    if not parts:
        parts.append({"text": ""})

    return {"role": gemini_role, "parts": parts}


def responses_request_to_gemini(
    request: ResponsesRequest,
) -> Dict[str, Any]:
    """
    Transform a Responses API request to Gemini format.

    Args:
        request: Responses API format request

    Returns:
        Dictionary in Gemini API format
    """
    contents: List[Dict[str, Any]] = []

    # Handle system instructions
    system_instruction = None
    if request.instructions:
        system_instruction = {"parts": [{"text": request.instructions}]}

    # Process input
    input_data = request.input
    if isinstance(input_data, str):
        # Simple string input
        contents.append({"role": "user", "parts": [{"text": input_data}]})
    elif isinstance(input_data, list):
        # Array of input items
        for item in input_data:
            if isinstance(item, dict):
                gemini_content = _process_input_item(item)
                if gemini_content:
                    contents.append(gemini_content)

    # Build generation config
    generation_config: Dict[str, Any] = {}

    if request.temperature is not None:
        generation_config["temperature"] = request.temperature
    if request.top_p is not None:
        generation_config["topP"] = request.top_p
    if request.max_output_tokens is not None:
        generation_config["maxOutputTokens"] = request.max_output_tokens

    # Build request payload
    request_payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": get_base_model_name(request.model),
    }

    if system_instruction:
        request_payload["systemInstruction"] = system_instruction

    # Handle tools
    tools_list: List[Dict[str, Any]] = []
    has_web_search = False

    # Add function declarations from tools
    if request.tools:
        function_declarations, web_search = _transform_responses_tools_to_gemini(
            request.tools
        )
        if web_search and not has_web_search:
            tools_list.append({"googleSearch": {}})
        if function_declarations:
            tools_list.append({"functionDeclarations": function_declarations})

    if tools_list:
        request_payload["tools"] = tools_list

    # Add tool choice config
    if request.tool_choice:
        tool_config = _transform_tool_choice_to_gemini(request.tool_choice)
        if tool_config:
            request_payload["toolConfig"] = {"functionCallingConfig": tool_config}

    # Add thinking config
    thinking_config = _build_thinking_config(request.model, request.reasoning)
    if thinking_config:
        request_payload["generationConfig"]["thinkingConfig"] = thinking_config

    return request_payload


def _extract_content_and_function_calls(
    parts: List[Dict[str, Any]],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Extract content, reasoning, and function calls from Gemini response parts.

    Args:
        parts: List of Gemini content parts

    Returns:
        Tuple of (content, reasoning_content, function_calls)
    """
    content_parts: List[str] = []
    reasoning_content = ""
    function_calls: List[Dict[str, Any]] = []

    for part in parts:
        # Text parts
        if part.get("text") is not None:
            if part.get("thought", False):
                reasoning_content += part.get("text", "")
            else:
                content_parts.append(part.get("text", ""))
            continue

        # Function call parts
        if part.get("functionCall"):
            func_call = part["functionCall"]
            call_id = f"call_{uuid.uuid4().hex[:24]}"
            function_calls.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:16]}",
                    "call_id": call_id,
                    "name": func_call.get("name", ""),
                    "arguments": json.dumps(func_call.get("args", {})),
                    "status": "completed",
                }
            )

        # Inline image data
        inline = part.get("inlineData")
        if inline and inline.get("data"):
            mime = inline.get("mimeType") or "image/png"
            if isinstance(mime, str) and mime.startswith("image/"):
                data_b64 = inline.get("data")
                content_parts.append(f"![image](data:{mime};base64,{data_b64})")

    content = "\n\n".join(p for p in content_parts if p)
    return content, reasoning_content, function_calls


def gemini_response_to_responses(
    gemini_response: Dict[str, Any], model: str
) -> Dict[str, Any]:
    """
    Transform a Gemini API response to Responses API format.

    Args:
        gemini_response: Response from Gemini API
        model: Model name to include in response

    Returns:
        Dictionary in Responses API format
    """
    response_id = f"resp_{uuid.uuid4().hex}"
    output: List[Dict[str, Any]] = []
    output_text = ""

    for candidate in gemini_response.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])
        content, reasoning_content, function_calls = (
            _extract_content_and_function_calls(parts)
        )

        # Add reasoning item if present
        if reasoning_content:
            output.append(
                {
                    "id": f"rs_{uuid.uuid4().hex[:32]}",
                    "type": "reasoning",
                    "content": [{"type": "text", "text": reasoning_content}],
                    "summary": [],
                }
            )

        # Add function calls
        for fc in function_calls:
            output.append(fc)

        # Add message output if there's content
        if content:
            output_text = content
            output.append(
                {
                    "id": f"msg_{uuid.uuid4().hex[:32]}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": [],
                        }
                    ],
                }
            )

    # Build usage info from Gemini response
    usage_metadata = gemini_response.get("usageMetadata", {})
    usage = {
        "input_tokens": usage_metadata.get("promptTokenCount", 0),
        "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
        "total_tokens": usage_metadata.get("totalTokenCount", 0),
    }

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "output": output,
        "output_text": output_text if output_text else None,
        "usage": usage,
        "status": "completed",
    }


def gemini_stream_chunk_to_responses_events(
    gemini_chunk: Dict[str, Any], model: str, response_id: str, output_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Transform a Gemini streaming chunk to Responses API streaming events.

    Args:
        gemini_chunk: Single chunk from Gemini streaming response
        model: Model name
        response_id: Response ID for this stream
        output_index: Current output item index

    Returns:
        List of Responses API streaming events
    """
    events: List[Dict[str, Any]] = []

    for candidate in gemini_chunk.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])
        content, reasoning_content, function_calls = (
            _extract_content_and_function_calls(parts)
        )

        # Emit function call events
        for fc in function_calls:
            # Item added event
            events.append(
                {
                    "type": "response.output_item.added",
                    "response_id": response_id,
                    "output_index": output_index,
                    "item": fc,
                }
            )
            # Arguments done event
            events.append(
                {
                    "type": "response.function_call_arguments.done",
                    "response_id": response_id,
                    "output_index": output_index,
                    "item": fc,
                }
            )
            output_index += 1

        # Emit content delta events
        if content:
            events.append(
                {
                    "type": "response.output_text.delta",
                    "response_id": response_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": content,
                }
            )

        # Emit reasoning delta if present
        if reasoning_content:
            events.append(
                {
                    "type": "response.reasoning.delta",
                    "response_id": response_id,
                    "output_index": output_index,
                    "delta": reasoning_content,
                }
            )

    return events
