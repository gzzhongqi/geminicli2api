"""
Gemini API Routes - Handles native Gemini API endpoints.
"""

import json
import logging
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from ..services.auth import authenticate_user
from ..services.gemini_client import (
    send_gemini_request,
    build_gemini_payload_from_native,
)
from ..models import SUPPORTED_MODELS
from ..config import create_error_response

logger = logging.getLogger(__name__)
router = APIRouter()


def _create_json_error_response(message: str, status_code: int) -> Response:
    """Create a JSON error response."""
    return Response(
        content=json.dumps(create_error_response(message, "api_error", status_code)),
        status_code=status_code,
        media_type="application/json",
    )


def _extract_model_from_path(path: str) -> Optional[str]:
    """
    Extract the model name from a Gemini API path.

    Examples:
    - "v1beta/models/gemini-1.5-pro/generateContent" -> "gemini-1.5-pro"
    - "v1/models/gemini-2.0-flash:streamGenerateContent" -> "gemini-2.0-flash"

    Args:
        path: The API path

    Returns:
        Model name or None if not found
    """
    parts = path.split("/")

    try:
        models_index = parts.index("models")
        if models_index + 1 < len(parts):
            model_name = parts[models_index + 1]
            # Remove action suffix like ":streamGenerateContent"
            if ":" in model_name:
                model_name = model_name.split(":")[0]
            return model_name
    except ValueError:
        pass

    return None


@router.get(
    "/v1beta/models",
    tags=["Gemini Native"],
    summary="List Gemini models",
    description="""
List all available Gemini models in native format.

**No authentication required** for this endpoint.
""",
)
async def gemini_list_models(request: Request) -> Response:
    """Native Gemini models endpoint. No authentication required."""
    try:
        logger.info("Gemini models list requested")
        return Response(
            content=json.dumps({"models": SUPPORTED_MODELS}),
            status_code=200,
            media_type="application/json; charset=utf-8",
        )
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return _create_json_error_response(f"Failed to list models: {e}", 500)


@router.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    response_model=None,
    tags=["Gemini Native"],
    summary="Gemini API proxy",
    description="""
Native Gemini API passthrough proxy.

Forwards requests directly to the Gemini API. Supports all native Gemini endpoints:

- `POST /v1beta/models/{model}:generateContent` - Generate content
- `POST /v1beta/models/{model}:streamGenerateContent` - Stream generate content

The request body should be in native Gemini format.
""",
)
async def gemini_proxy(
    request: Request,
    full_path: str,
    username: str = Depends(authenticate_user),
) -> Union[Response, StreamingResponse]:
    """
    Native Gemini API proxy endpoint.

    Handles paths like:
    - /v1beta/models/{model}:generateContent
    - /v1beta/models/{model}:streamGenerateContent
    """
    try:
        post_data = await request.body()
        is_streaming = "stream" in full_path.lower()
        model_name = _extract_model_from_path(full_path)

        logger.info(
            f"Gemini proxy: path={full_path}, model={model_name}, stream={is_streaming}"
        )

        if not model_name:
            logger.error(f"Could not extract model from path: {full_path}")
            return _create_json_error_response(
                f"Could not extract model name from path: {full_path}", 400
            )

        # Parse request body
        try:
            incoming_request: Dict[str, Any] = (
                json.loads(post_data) if post_data else {}
            )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return _create_json_error_response("Invalid JSON in request body", 400)

        # Build and send request
        gemini_payload = build_gemini_payload_from_native(incoming_request, model_name)
        response = send_gemini_request(gemini_payload, is_streaming=is_streaming)

        if hasattr(response, "status_code"):
            if response.status_code != 200:
                logger.error(f"Gemini API error: {response.status_code}")
            else:
                logger.info(f"Processed request for model: {model_name}")

        return response

    except Exception as e:
        logger.error(f"Proxy error: {e}")
        return _create_json_error_response(f"Proxy error: {e}", 500)
