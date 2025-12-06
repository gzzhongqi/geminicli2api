"""
OpenAI Responses API Routes - Handles the newer Responses API endpoints.
"""

import json
import uuid
import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Union

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from ..services.auth import authenticate_user
from ..services.gemini_client import (
    send_gemini_request,
    build_gemini_payload_from_openai,
)
from ..schemas import ResponsesRequest
from ..config import create_error_response
from .transformers import (
    responses_request_to_gemini,
    gemini_response_to_responses,
    gemini_stream_chunk_to_responses_events,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _create_error_response(
    message: str, status_code: int, error_type: str = "api_error"
) -> Response:
    """Create a JSON error response."""
    return Response(
        content=json.dumps(create_error_response(message, error_type, status_code)),
        status_code=status_code,
        media_type="application/json",
    )


def _parse_response_body(response: Response) -> Dict[str, Any]:
    """Parse response body to dict."""
    body = response.body
    if isinstance(body, bytes):
        body_str = body.decode("utf-8", "ignore")
    elif isinstance(body, memoryview):
        body_str = bytes(body).decode("utf-8", "ignore")
    else:
        body_str = str(body)
    return json.loads(body_str)


async def _stream_responses_response(
    gemini_payload: Dict[str, Any], model: str
) -> AsyncGenerator[str, None]:
    """
    Generate Responses API formatted streaming response from Gemini.

    Args:
        gemini_payload: Prepared Gemini request payload
        model: Model name for response

    Yields:
        SSE formatted strings with Responses API events
    """
    response_id = f"resp_{uuid.uuid4().hex}"
    output_index = 0

    try:
        response = send_gemini_request(gemini_payload, is_streaming=True)

        if not isinstance(response, StreamingResponse):
            # Handle error response
            error_msg = "Streaming request failed"
            status_code = getattr(response, "status_code", 500)

            if hasattr(response, "body"):
                try:
                    error_data = _parse_response_body(response)
                    if "error" in error_data:
                        error_msg = error_data["error"].get("message", error_msg)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            logger.error(f"Streaming failed: {error_msg}")
            error_event = {
                "type": "error",
                "error": {"message": error_msg, "code": status_code},
            }
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        logger.info(f"Starting Responses stream: {response_id}")

        # Send response.created event
        created_event = {
            "type": "response.created",
            "response_id": response_id,
            "model": model,
        }
        yield f"event: response.created\ndata: {json.dumps(created_event)}\n\n"

        accumulated_text = ""
        accumulated_function_calls = []

        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8", "ignore")
            elif isinstance(chunk, memoryview):
                chunk_str = bytes(chunk).decode("utf-8", "ignore")
            else:
                chunk_str = str(chunk)

            if not chunk_str.startswith("data: "):
                continue

            try:
                gemini_chunk = json.loads(chunk_str[6:])

                # Handle error chunk
                if "error" in gemini_chunk:
                    error = gemini_chunk["error"]
                    logger.error(f"Stream error: {error}")
                    error_event = {
                        "type": "error",
                        "error": error,
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                    yield "event: done\ndata: {}\n\n"
                    return

                # Transform to Responses events
                events = gemini_stream_chunk_to_responses_events(
                    gemini_chunk, model, response_id, output_index
                )

                for event in events:
                    event_type = event.get("type", "response.output_text.delta")
                    yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

                    # Track output index for function calls
                    if event_type == "response.function_call_arguments.done":
                        output_index += 1
                        accumulated_function_calls.append(event.get("item"))
                    elif event_type == "response.output_text.delta":
                        accumulated_text += event.get("delta", "")

                await asyncio.sleep(0)

            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to parse chunk: {e}")
                continue

        # Send response.completed event
        output = []
        if accumulated_function_calls:
            output.extend(accumulated_function_calls)
        if accumulated_text:
            output.append(
                {
                    "id": f"msg_{uuid.uuid4().hex[:32]}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": accumulated_text,
                            "annotations": [],
                        }
                    ],
                }
            )

        completed_event = {
            "type": "response.completed",
            "response_id": response_id,
            "model": model,
            "output": output,
            "output_text": accumulated_text if accumulated_text else None,
        }
        yield f"event: response.completed\ndata: {json.dumps(completed_event)}\n\n"
        yield "event: done\ndata: {}\n\n"

        logger.info(f"Completed Responses stream: {response_id}")

    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_event = {"type": "error", "error": {"message": str(e)}}
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
        yield "event: done\ndata: {}\n\n"


def _handle_non_streaming_response(
    gemini_payload: Dict[str, Any], model: str
) -> Union[Dict[str, Any], Response]:
    """
    Handle non-streaming Gemini request and response for Responses API.

    Args:
        gemini_payload: Prepared Gemini request payload
        model: Model name for response

    Returns:
        Responses API formatted response dict or error Response
    """
    response = send_gemini_request(gemini_payload, is_streaming=False)

    # Handle error responses
    if isinstance(response, Response) and response.status_code != 200:
        logger.error(f"Gemini API error: {response.status_code}")

        try:
            error_data = _parse_response_body(response)
            if "error" in error_data:
                error = error_data["error"]
                error_type = (
                    "invalid_request_error"
                    if response.status_code == 404
                    else "api_error"
                )
                return _create_error_response(
                    error.get("message", f"API error: {response.status_code}"),
                    response.status_code,
                    error.get("type", error_type),
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        error_type = (
            "invalid_request_error" if response.status_code == 404 else "api_error"
        )
        return _create_error_response(
            f"API error: {response.status_code}", response.status_code, error_type
        )

    # Parse and transform response
    try:
        gemini_response = _parse_response_body(response)
        responses_response = gemini_response_to_responses(gemini_response, model)
        logger.info(f"Processed Responses API response for model: {model}")
        return responses_response
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse response: {e}")
        return _create_error_response(f"Failed to process response: {e}", 500)


@router.post(
    "/v1/responses",
    response_model=None,
    tags=["OpenAI Compatible"],
    summary="Create response (Responses API)",
    description="""
Create a response using the OpenAI Responses API format.

This is a newer, simpler API that uses 'input' instead of 'messages' and returns
output items that can include messages, function calls, and reasoning.

**Models:** Use Gemini model names directly (e.g., `gemini-2.5-pro`, `gemini-2.5-flash`).

**Features:**
- Simple string or array input
- System instructions via `instructions` parameter
- Function calling with simplified tool definitions
- Web search with `{ "type": "web_search" }` tool
- Streaming with Server-Sent Events

**Streaming:** Set `stream: true` for SSE streaming with Responses API events.
""",
)
async def responses_create(
    request: ResponsesRequest,
    username: str = Depends(authenticate_user),
) -> Union[Dict[str, Any], Response, StreamingResponse]:
    """OpenAI Responses API endpoint."""
    try:
        logger.info(f"Responses API: model={request.model}, stream={request.stream}")
        gemini_request_data = responses_request_to_gemini(request)
        gemini_payload = build_gemini_payload_from_openai(gemini_request_data)
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        return _create_error_response(
            f"Request processing failed: {e}", 400, "invalid_request_error"
        )

    if request.stream:
        return StreamingResponse(
            _stream_responses_response(gemini_payload, request.model),
            media_type="text/event-stream",
        )

    try:
        return _handle_non_streaming_response(gemini_payload, request.model)
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return _create_error_response(f"Request failed: {e}", 500)
