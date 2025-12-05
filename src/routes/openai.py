"""
OpenAI API Routes - Handles OpenAI-compatible endpoints.
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
from ..schemas import ChatCompletionRequest
from ..models import SUPPORTED_MODELS
from ..config import MODEL_CREATED_TIMESTAMP, create_error_response
from .transformers import (
    openai_request_to_gemini,
    gemini_response_to_openai,
    gemini_stream_chunk_to_openai,
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


async def _stream_openai_response(
    gemini_payload: Dict[str, Any], model: str
) -> AsyncGenerator[str, None]:
    """
    Generate OpenAI-formatted streaming response from Gemini.

    Args:
        gemini_payload: Prepared Gemini request payload
        model: Model name for response

    Yields:
        SSE formatted strings
    """
    response_id = f"chatcmpl-{uuid.uuid4()}"

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
            error_type = "invalid_request_error" if status_code == 404 else "api_error"
            yield f"data: {json.dumps(create_error_response(error_msg, error_type, status_code))}\n\n"
            yield "data: [DONE]\n\n"
            return

        logger.info(f"Starting stream: {response_id}")

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
                    yield f"data: {json.dumps(create_error_response(error.get('message', 'Unknown error'), error.get('type', 'api_error'), error.get('code')))}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Transform and yield
                openai_chunk = gemini_stream_chunk_to_openai(
                    gemini_chunk, model, response_id
                )
                yield f"data: {json.dumps(openai_chunk)}\n\n"
                await asyncio.sleep(0)

            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to parse chunk: {e}")
                continue

        yield "data: [DONE]\n\n"
        logger.info(f"Completed stream: {response_id}")

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps(create_error_response(f'Streaming failed: {e}', 'api_error', 500))}\n\n"
        yield "data: [DONE]\n\n"


def _handle_non_streaming_response(
    gemini_payload: Dict[str, Any], model: str
) -> Union[Dict[str, Any], Response]:
    """
    Handle non-streaming Gemini request and response.

    Args:
        gemini_payload: Prepared Gemini request payload
        model: Model name for response

    Returns:
        OpenAI-formatted response dict or error Response
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
        openai_response = gemini_response_to_openai(gemini_response, model)
        logger.info(f"Processed response for model: {model}")
        return openai_response
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse response: {e}")
        return _create_error_response(f"Failed to process response: {e}", 500)


@router.post(
    "/v1/chat/completions",
    response_model=None,
    tags=["OpenAI Compatible"],
    summary="Create chat completion",
    description="""
Create a chat completion using OpenAI-compatible format.

**Model Mapping:**
- `gpt-4o`, `gpt-4` → `gemini-2.0-flash`
- `gpt-4o-mini`, `gpt-3.5-turbo` → `gemini-2.0-flash-lite`  
- `o1`, `o1-pro` → `gemini-2.5-pro`
- `o3`, `o3-mini` → `gemini-2.5-flash`

**Streaming:** Set `stream: true` for Server-Sent Events (SSE) streaming.

**Thinking Models:** Use `o1` or `o3` series models with `reasoning_effort` parameter
for extended thinking capabilities.
""",
)
async def openai_chat_completions(
    request: ChatCompletionRequest,
    username: str = Depends(authenticate_user),
) -> Union[Dict[str, Any], Response, StreamingResponse]:
    """OpenAI-compatible chat completions endpoint."""
    try:
        logger.info(f"Chat completion: model={request.model}, stream={request.stream}")
        gemini_request_data = openai_request_to_gemini(request)
        gemini_payload = build_gemini_payload_from_openai(gemini_request_data)
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        return _create_error_response(
            f"Request processing failed: {e}", 400, "invalid_request_error"
        )

    if request.stream:
        return StreamingResponse(
            _stream_openai_response(gemini_payload, request.model),
            media_type="text/event-stream",
        )

    try:
        return _handle_non_streaming_response(gemini_payload, request.model)
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return _create_error_response(f"Request failed: {e}", 500)


@router.get(
    "/v1/models",
    response_model=None,
    tags=["OpenAI Compatible"],
    summary="List available models",
    description="""
List all available Gemini models in OpenAI-compatible format.

**No authentication required** for this endpoint.

Returns model information including capabilities and permissions.
""",
)
async def openai_list_models() -> Union[Dict[str, Any], Response]:
    """OpenAI-compatible models endpoint. No authentication required."""
    try:
        logger.info("Models list requested")

        openai_models = []
        for model in SUPPORTED_MODELS:
            model_id = model["name"].replace("models/", "")
            openai_models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": MODEL_CREATED_TIMESTAMP,
                    "owned_by": "google",
                    "permission": [
                        {
                            "id": f"modelperm-{model_id.replace('/', '-')}",
                            "object": "model_permission",
                            "created": MODEL_CREATED_TIMESTAMP,
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": False,
                            "allow_search_indices": False,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False,
                        }
                    ],
                    "root": model_id,
                    "parent": None,
                }
            )

        logger.info(f"Returning {len(openai_models)} models")
        return {"object": "list", "data": openai_models}

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return _create_error_response(f"Failed to list models: {e}", 500)
