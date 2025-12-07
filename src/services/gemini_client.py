"""
Google Gemini API Client - Handles all communication with Google's Gemini API.
Supports multiple credentials with automatic fallback on failure.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Union

import requests
from fastapi import Response
from fastapi.responses import StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials

from .auth import (
    get_credentials,
    save_credentials,
    get_user_project_id,
    discover_project_id_for_credential,
    onboard_user,
    get_next_credential,
    get_fallback_credential,
    mark_credential_failed,
    mark_credential_success,
    get_credential_project_id,
    set_credential_project_id,
    is_credential_onboarded,
    set_credential_onboarded,
    get_pool_stats,
)
from ..utils import get_user_agent
from ..models import (
    get_base_model_name,
    should_include_thoughts,
)
from ..config import (
    CODE_ASSIST_ENDPOINT,
    DEFAULT_SAFETY_SETTINGS,
    STREAMING_RESPONSE_HEADERS,
    create_error_response,
)

logger = logging.getLogger(__name__)

# Error codes that should trigger credential fallback
FALLBACK_ERROR_CODES = {401, 403, 429}


def _create_json_error_response(message: str, status_code: int) -> Response:
    """Create a JSON error response."""
    error_type = "invalid_request_error" if status_code == 404 else "api_error"
    return Response(
        content=json.dumps(create_error_response(message, error_type, status_code)),
        status_code=status_code,
        media_type="application/json",
    )


async def _error_stream_generator(
    message: str, status_code: int
) -> AsyncGenerator[bytes, None]:
    """Generate a streaming error response."""
    error_type = "invalid_request_error" if status_code == 404 else "api_error"
    error_data = create_error_response(message, error_type, status_code)
    yield f"data: {json.dumps(error_data)}\n\n".encode("utf-8")


async def _stream_generator(resp: requests.Response) -> AsyncGenerator[bytes, None]:
    """
    Stream response chunks from Google API.

    Args:
        resp: Response object from requests library

    Yields:
        Encoded SSE data chunks
    """
    try:
        with resp:
            for chunk in resp.iter_lines():
                if not chunk:
                    continue

                if not isinstance(chunk, str):
                    chunk = chunk.decode("utf-8", "ignore")

                if not chunk.startswith("data: "):
                    continue

                chunk_data = chunk[6:]  # Remove 'data: ' prefix

                try:
                    obj = json.loads(chunk_data)

                    if "response" in obj:
                        response_chunk = obj["response"]
                        response_json = json.dumps(
                            response_chunk, separators=(",", ":")
                        )
                        yield f"data: {response_json}\n\n".encode("utf-8")
                    else:
                        obj_json = json.dumps(obj, separators=(",", ":"))
                        yield f"data: {obj_json}\n\n".encode("utf-8")

                    await asyncio.sleep(0)

                except json.JSONDecodeError:
                    continue

    except requests.exceptions.RequestException as e:
        logger.error(f"Streaming request failed: {e}")
        error_data = create_error_response(
            f"Upstream request failed: {e}", "api_error", 502
        )
        yield f"data: {json.dumps(error_data)}\n\n".encode("utf-8")
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_data = create_error_response(f"Unexpected error: {e}", "api_error", 500)
        yield f"data: {json.dumps(error_data)}\n\n".encode("utf-8")


def _handle_streaming_response(resp: requests.Response) -> StreamingResponse:
    """Handle streaming response from Google API."""
    if resp.status_code != 200:
        logger.error(f"Google API error {resp.status_code}: {resp.text}")
        error_message = f"Google API error: {resp.status_code}"

        try:
            error_data = resp.json()
            if "error" in error_data:
                error_message = error_data["error"].get("message", error_message)
        except (json.JSONDecodeError, ValueError):
            pass

        return StreamingResponse(
            _error_stream_generator(error_message, resp.status_code),
            media_type="text/event-stream",
            headers=STREAMING_RESPONSE_HEADERS,
            status_code=resp.status_code,
        )

    return StreamingResponse(
        _stream_generator(resp),
        media_type="text/event-stream",
        headers=STREAMING_RESPONSE_HEADERS,
    )


def _handle_non_streaming_response(resp: requests.Response) -> Response:
    """Handle non-streaming response from Google API."""
    if resp.status_code == 200:
        try:
            response_text = resp.text
            if response_text.startswith("data: "):
                response_text = response_text[6:]

            google_response = json.loads(response_text)
            standard_response = google_response.get("response")

            return Response(
                content=json.dumps(standard_response),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse response: {e}")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("Content-Type"),
            )

    # Handle error response
    logger.error(f"Google API error {resp.status_code}: {resp.text}")

    try:
        error_data = resp.json()
        if "error" in error_data:
            error_message = error_data["error"].get(
                "message", f"API error: {resp.status_code}"
            )
            return _create_json_error_response(error_message, resp.status_code)
    except (json.JSONDecodeError, KeyError):
        pass

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("Content-Type"),
    )


def send_gemini_request(
    payload: Dict[str, Any], is_streaming: bool = False
) -> Union[Response, StreamingResponse]:
    """
    Send a request to Google's Gemini API with multi-credential support.

    Uses round-robin credential selection and automatic fallback on failure.

    Args:
        payload: The request payload in Gemini format
        is_streaming: Whether this is a streaming request

    Returns:
        FastAPI Response or StreamingResponse object
    """
    # Get credential from pool
    cred_result = get_next_credential()
    if not cred_result:
        return _create_json_error_response(
            "Authentication failed. Please restart the proxy to log in.", 500
        )

    creds, cred_index = cred_result

    # Log credential usage for multi-credential mode
    pool_stats = get_pool_stats()
    if pool_stats["total"] > 1:
        logger.info(
            f"Using credential #{cred_index + 1} of {pool_stats['total']} "
            f"({pool_stats['available']} available)"
        )

    # Try to send request with current credential
    response = _send_request_with_credential(payload, creds, cred_index, is_streaming)

    # Check if we should try fallback
    if _should_try_fallback(response, cred_index, pool_stats):
        logger.warning(f"Credential #{cred_index + 1} failed, trying fallback...")
        mark_credential_failed(cred_index)

        fallback_result = get_fallback_credential(cred_index)
        if fallback_result:
            fallback_creds, fallback_index = fallback_result
            logger.info(f"Retrying with fallback credential #{fallback_index + 1}")
            response = _send_request_with_credential(
                payload, fallback_creds, fallback_index, is_streaming
            )

            if not _is_error_response(response):
                mark_credential_success(fallback_index)
    elif not _is_error_response(response):
        mark_credential_success(cred_index)

    return response


def _should_try_fallback(
    response: Union[Response, StreamingResponse], cred_index: int, pool_stats: dict
) -> bool:
    """
    Determine if we should try a fallback credential.

    Args:
        response: The response from the first attempt
        cred_index: Index of the credential used
        pool_stats: Credential pool statistics

    Returns:
        True if fallback should be attempted
    """
    # Only try fallback if we have multiple credentials
    if pool_stats["total"] <= 1:
        return False

    # Single credential mode doesn't support fallback
    if cred_index == -1:
        return False

    # Check if response indicates an error that warrants fallback
    if isinstance(response, Response):
        return response.status_code in FALLBACK_ERROR_CODES
    elif isinstance(response, StreamingResponse):
        return response.status_code in FALLBACK_ERROR_CODES

    return False


def _is_error_response(response: Union[Response, StreamingResponse]) -> bool:
    """Check if response is an error."""
    if isinstance(response, (Response, StreamingResponse)):
        return response.status_code >= 400
    return False


def _send_request_with_credential(
    payload: Dict[str, Any],
    creds: Credentials,
    cred_index: int,
    is_streaming: bool,
) -> Union[Response, StreamingResponse]:
    """
    Send a request using a specific credential.

    Args:
        payload: The request payload
        creds: Credentials to use
        cred_index: Index of the credential (-1 for single credential mode)
        is_streaming: Whether this is a streaming request

    Returns:
        Response or StreamingResponse
    """
    # Refresh credentials if needed
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            if cred_index == -1:
                save_credentials(creds)
            logger.debug(f"Refreshed credential #{cred_index + 1}")
        except Exception as e:
            logger.error(f"Token refresh failed for credential #{cred_index + 1}: {e}")
            return _create_json_error_response(
                "Token refresh failed. Please restart the proxy to re-authenticate.",
                500,
            )
    elif not creds.token:
        return _create_json_error_response(
            "No access token. Please restart the proxy to re-authenticate.", 500
        )

    # Get project ID
    proj_id = _get_project_id_for_credential(creds, cred_index)
    if not proj_id:
        return _create_json_error_response("Failed to get user project ID.", 500)

    # Onboard user if needed
    if not is_credential_onboarded(cred_index):
        try:
            onboard_user(creds, proj_id)
            set_credential_onboarded(cred_index)
        except Exception as e:
            logger.error(f"Onboarding failed for credential #{cred_index + 1}: {e}")
            return _create_json_error_response(f"Onboarding failed: {e}", 500)

    # Build final payload
    final_payload = {
        "model": payload.get("model"),
        "project": proj_id,
        "request": payload.get("request", {}),
    }

    # Determine URL
    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    try:
        if is_streaming:
            resp = requests.post(
                target_url,
                data=json.dumps(final_payload),
                headers=headers,
                stream=True,
            )
            return _handle_streaming_response(resp)
        else:
            resp = requests.post(
                target_url,
                data=json.dumps(final_payload),
                headers=headers,
            )
            return _handle_non_streaming_response(resp)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return _create_json_error_response(f"Request failed: {e}", 500)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return _create_json_error_response(f"Unexpected error: {e}", 500)


def _get_project_id_for_credential(
    creds: Credentials, cred_index: int
) -> Optional[str]:
    """
    Get project ID for a specific credential.

    Args:
        creds: The credentials
        cred_index: Credential index

    Returns:
        Project ID or None
    """
    # Check if credential has cached project ID
    cached_proj_id = get_credential_project_id(cred_index)
    if cached_proj_id:
        return cached_proj_id

    # For multi-credential mode, discover project ID for this specific credential
    if cred_index != -1:
        try:
            proj_id = discover_project_id_for_credential(creds)
            if proj_id:
                set_credential_project_id(cred_index, proj_id)
            return proj_id
        except Exception as e:
            logger.error(
                f"Failed to get project ID for credential #{cred_index + 1}: {e}"
            )
            return None

    # Fallback to global project ID discovery (single credential mode)
    try:
        proj_id = get_user_project_id(creds)
        return proj_id
    except Exception as e:
        logger.error(f"Failed to get project ID: {e}")
        return None


def build_gemini_payload_from_openai(openai_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a Gemini API payload from an OpenAI-transformed request.

    Args:
        openai_payload: Payload already transformed from OpenAI format

    Returns:
        Payload ready for Google API
    """
    request_data = {
        "contents": openai_payload.get("contents"),
        "systemInstruction": openai_payload.get("systemInstruction"),
        "cachedContent": openai_payload.get("cachedContent"),
        "tools": openai_payload.get("tools"),
        "toolConfig": openai_payload.get("toolConfig"),
        "safetySettings": openai_payload.get("safetySettings", DEFAULT_SAFETY_SETTINGS),
        "generationConfig": openai_payload.get("generationConfig", {}),
    }

    # Remove None values
    request_data = {k: v for k, v in request_data.items() if v is not None}

    return {
        "model": openai_payload.get("model"),
        "request": request_data,
    }


def build_gemini_payload_from_native(
    native_request: Dict[str, Any], model_from_path: str
) -> Dict[str, Any]:
    """
    Build a Gemini API payload from a native Gemini request.

    Args:
        native_request: Native Gemini API request
        model_from_path: Model name extracted from URL path

    Returns:
        Payload ready for Google API
    """
    native_request["safetySettings"] = DEFAULT_SAFETY_SETTINGS

    if "generationConfig" not in native_request:
        native_request["generationConfig"] = {}

    if "thinkingConfig" not in native_request["generationConfig"]:
        native_request["generationConfig"]["thinkingConfig"] = {}

    # Configure thinking for non-image models
    if "gemini-2.5-flash-image" not in model_from_path:
        include_thoughts = should_include_thoughts(model_from_path)

        thinking_config = native_request["generationConfig"]["thinkingConfig"]
        thinking_config["includeThoughts"] = include_thoughts

        # Use auto thinking budget if not specified
        if "thinkingBudget" not in thinking_config:
            thinking_config["thinkingBudget"] = -1

    return {
        "model": get_base_model_name(model_from_path),
        "request": native_request,
    }
