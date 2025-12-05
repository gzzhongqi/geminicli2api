"""
Anthropic API Routes - Handles Anthropic/Claude-compatible endpoints.
This module provides Anthropic-compatible endpoints that transform requests/responses
and delegate to the Google API client.
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..config import GEMINI_AUTH_PASSWORD
from ..schemas import AnthropicMessagesRequest
from ..services.gemini_client import (
    build_gemini_payload_from_openai,
    send_gemini_request,
)
from .transformers import (
    AnthropicStreamProcessor,
    anthropic_request_to_gemini,
    create_anthropic_error,
    format_sse_event,
    gemini_response_to_anthropic,
)

router = APIRouter()


def authenticate_anthropic(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
) -> str:
    """
    Authenticate using either Authorization header or x-api-key header.
    Anthropic API typically uses x-api-key, but we also support Authorization.

    Args:
        authorization: Authorization header (Bearer token or API key)
        x_api_key: Anthropic-style x-api-key header

    Returns:
        Username/identifier string
    """
    # Try x-api-key first (Anthropic style)
    if x_api_key:
        if x_api_key == GEMINI_AUTH_PASSWORD:
            return "anthropic_user"
        # Also accept "sk-" prefixed keys where the key follows
        if x_api_key.startswith("sk-") and x_api_key[3:] == GEMINI_AUTH_PASSWORD:
            return "anthropic_user"

    # Fall back to Authorization header
    if authorization:
        # Handle "Bearer <token>" format
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            if token == GEMINI_AUTH_PASSWORD:
                return "anthropic_user"
        # Handle plain token
        elif authorization == GEMINI_AUTH_PASSWORD:
            return "anthropic_user"

    # If no valid auth found, raise error
    raise HTTPException(
        status_code=401,
        detail={
            "type": "authentication_error",
            "message": "Invalid API key. Provide via x-api-key header or Authorization: Bearer <key>",
        },
    )


@router.post(
    "/v1/messages",
    tags=["Anthropic Compatible"],
    summary="Create a message",
    description="""
Create a message using Anthropic Claude-compatible format.

**Model Mapping:** All Claude models are mapped to `gemini-2.5-flash`.

**Authentication:** Requires either:
- `x-api-key` header (Anthropic style)
- `Authorization: Bearer <token>` header

**Streaming:** Set `stream: true` for Server-Sent Events (SSE) streaming
with Anthropic-style events (`message_start`, `content_block_delta`, etc.).

**Extended Thinking:** Enable with `thinking: {type: "enabled", budget_tokens: 4096}`
to receive model reasoning in the response.

**Tool Use:** Define tools in the `tools` array and receive `tool_use` content blocks
when the model wants to call a tool.
""",
)
async def anthropic_messages(
    request: AnthropicMessagesRequest,
    http_request: Request,
    username: str = Depends(authenticate_anthropic),
):
    """
    Anthropic-compatible messages endpoint.
    Transforms Anthropic requests to Gemini format, sends to Google API,
    and transforms responses back to Anthropic format.

    Supports both streaming and non-streaming modes.
    """

    try:
        logging.info(
            f"Anthropic messages request: model={request.model}, stream={request.stream}"
        )

        # Determine if thinking should be included in response
        # Only include if client explicitly requests it via thinking config
        include_thinking = (
            request.thinking is not None and request.thinking.type == "enabled"
        )

        # Transform Anthropic request to Gemini format
        gemini_request_data = anthropic_request_to_gemini(request)

        # Build the payload for Google API
        gemini_payload = build_gemini_payload_from_openai(gemini_request_data)

    except Exception as e:
        logging.error(f"Error processing Anthropic request: {str(e)}")
        return Response(
            content=json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Request processing failed: {str(e)}",
                    },
                }
            ),
            status_code=400,
            media_type="application/json",
        )

    if request.stream:
        # Handle streaming response
        async def anthropic_stream_generator():
            try:
                response = send_gemini_request(gemini_payload, is_streaming=True)

                if isinstance(response, StreamingResponse):
                    processor = AnthropicStreamProcessor(
                        request.model, include_thinking
                    )
                    logging.info(
                        f"Starting Anthropic streaming response: {processor.message_id}"
                    )

                    async for chunk in response.body_iterator:
                        if isinstance(chunk, bytes):
                            chunk_str = chunk.decode("utf-8", "ignore")
                        elif isinstance(chunk, memoryview):
                            chunk_str = bytes(chunk).decode("utf-8", "ignore")
                        else:
                            chunk_str = str(chunk)

                        if chunk_str.startswith("data: "):
                            try:
                                # Parse the Gemini streaming chunk
                                chunk_data = chunk_str[6:]  # Remove 'data: ' prefix
                                gemini_chunk = json.loads(chunk_data)

                                # Check if this is an error chunk
                                if "error" in gemini_chunk:
                                    logging.error(
                                        f"Error in streaming response: {gemini_chunk['error']}"
                                    )
                                    error_event = create_anthropic_error(
                                        gemini_chunk["error"].get("type", "api_error"),
                                        gemini_chunk["error"].get(
                                            "message", "Unknown error"
                                        ),
                                    )
                                    yield format_sse_event(error_event)
                                    return

                                # Process chunk and emit Anthropic events
                                events = processor.process_chunk(gemini_chunk)
                                for event in events:
                                    yield format_sse_event(event)
                                    await asyncio.sleep(0)

                            except (
                                json.JSONDecodeError,
                                KeyError,
                                UnicodeDecodeError,
                            ) as e:
                                logging.warning(
                                    f"Failed to parse streaming chunk: {str(e)}"
                                )
                                continue

                    # Emit final events
                    final_events = processor.finalize()
                    for event in final_events:
                        yield format_sse_event(event)

                    logging.info(
                        f"Completed Anthropic streaming response: {processor.message_id}"
                    )
                else:
                    # Error case - handle Response object with error
                    error_msg = "Streaming request failed"

                    if hasattr(response, "body"):
                        try:
                            error_body = response.body
                            if isinstance(error_body, bytes):
                                error_body_str = error_body.decode("utf-8", "ignore")
                            elif isinstance(error_body, memoryview):
                                error_body_str = bytes(error_body).decode(
                                    "utf-8", "ignore"
                                )
                            else:
                                error_body_str = str(error_body)
                            error_data = json.loads(error_body_str)
                            if "error" in error_data:
                                error_msg = error_data["error"].get(
                                    "message", error_msg
                                )
                        except (json.JSONDecodeError, KeyError, TypeError):
                            pass

                    logging.error(f"Streaming request failed: {error_msg}")
                    error_event = create_anthropic_error("api_error", error_msg)
                    yield format_sse_event(error_event)

            except Exception as e:
                logging.error(f"Streaming error: {str(e)}")
                error_event = create_anthropic_error(
                    "api_error", f"Streaming failed: {str(e)}"
                )
                yield format_sse_event(error_event)

        return StreamingResponse(
            anthropic_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    else:
        # Handle non-streaming response
        try:
            response = send_gemini_request(gemini_payload, is_streaming=False)

            if isinstance(response, Response) and response.status_code != 200:
                # Handle error responses from Google API
                logging.error(f"Gemini API error: status={response.status_code}")

                error_msg = f"API error: {response.status_code}"
                error_type = "api_error"

                try:
                    error_body = response.body
                    if isinstance(error_body, bytes):
                        error_body_str = error_body.decode("utf-8", "ignore")
                    elif isinstance(error_body, memoryview):
                        error_body_str = bytes(error_body).decode("utf-8", "ignore")
                    else:
                        error_body_str = str(error_body)

                    error_data = json.loads(error_body_str)
                    if "error" in error_data:
                        error_msg = error_data["error"].get("message", error_msg)
                        if response.status_code == 404:
                            error_type = "not_found_error"
                        elif response.status_code == 400:
                            error_type = "invalid_request_error"
                        elif response.status_code == 429:
                            error_type = "rate_limit_error"
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

                return Response(
                    content=json.dumps(
                        {
                            "type": "error",
                            "error": {"type": error_type, "message": error_msg},
                        }
                    ),
                    status_code=response.status_code,
                    media_type="application/json",
                )

            try:
                # Parse Gemini response and transform to Anthropic format
                response_body = response.body
                if isinstance(response_body, bytes):
                    response_body_str = response_body.decode("utf-8", "ignore")
                elif isinstance(response_body, memoryview):
                    response_body_str = bytes(response_body).decode("utf-8", "ignore")
                else:
                    response_body_str = str(response_body)
                gemini_response = json.loads(response_body_str)
                anthropic_response = gemini_response_to_anthropic(
                    gemini_response, request.model, include_thinking=include_thinking
                )

                logging.info(
                    f"Successfully processed non-streaming Anthropic response for model: {request.model}"
                )
                return Response(
                    content=json.dumps(anthropic_response),
                    status_code=200,
                    media_type="application/json",
                )

            except (json.JSONDecodeError, AttributeError) as e:
                logging.error(f"Failed to parse Gemini response: {str(e)}")
                return Response(
                    content=json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": f"Failed to process response: {str(e)}",
                            },
                        }
                    ),
                    status_code=500,
                    media_type="application/json",
                )

        except Exception as e:
            logging.error(f"Non-streaming request failed: {str(e)}")
            return Response(
                content=json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"Request failed: {str(e)}",
                        },
                    }
                ),
                status_code=500,
                media_type="application/json",
            )
