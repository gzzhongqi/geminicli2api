"""
Configuration constants for the Geminicli2api proxy server.
Centralizes all configuration to avoid duplication across modules.
"""

import os
from typing import Any, Dict, List, Optional

# App Info
APP_VERSION = "1.0.0"
APP_NAME = "geminicli2api"

# API Endpoints
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

# OAuth URLs
TOKEN_URI = "https://oauth2.googleapis.com/token"
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
OAUTH_REDIRECT_URI = "http://localhost:8080"
OAUTH_CALLBACK_PORT = 8080

# Client Configuration
CLI_VERSION = "0.1.5"  # Match current gemini-cli version

# Timestamps
MODEL_CREATED_TIMESTAMP = 1677610602  # OpenAI model created timestamp

# Timeouts and Intervals
ONBOARD_POLL_INTERVAL = 5  # seconds
ONBOARD_MAX_RETRIES = 60  # max retries for onboarding (5 min total)

# Date Formats
ISO_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# OAuth Configuration
CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
CLIENT_SECRET = "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"
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# File Paths
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIAL_FILE = os.path.join(
    SCRIPT_DIR, os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "oauth_creds.json")
)

# Authentication
GEMINI_AUTH_PASSWORD = os.getenv("GEMINI_AUTH_PASSWORD", "123456")

# Default Safety Settings for Google API
DEFAULT_SAFETY_SETTINGS: List[Dict[str, str]] = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_UNSPECIFIED", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "BLOCK_NONE"},
]

# Streaming Response Headers
STREAMING_RESPONSE_HEADERS: Dict[str, str] = {
    "Content-Type": "text/event-stream",
    "Content-Disposition": "attachment",
    "Vary": "Origin, X-Origin, Referer",
    "Cache-Control": "no-cache, no-store, max-age=0, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "Mon, 01 Jan 1990 00:00:00 GMT",
    "X-Content-Type-Options": "nosniff",
}


def create_error_response(
    message: str, error_type: str = "api_error", code: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response dictionary.

    Args:
        message: Error message to display
        error_type: Type of error (e.g., "api_error", "invalid_request_error")
        code: Optional HTTP status code

    Returns:
        Standardized error response dictionary
    """
    error: Dict[str, Any] = {
        "message": message,
        "type": error_type,
    }
    if code is not None:
        error["code"] = code
    return {"error": error}
