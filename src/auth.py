"""
Authentication module for Geminicli2api.
Handles OAuth2 authentication with Google APIs.
"""

import os
import json
import base64
import time
import logging
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import urlparse, parse_qs

from fastapi import Request, HTTPException
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest
import requests

from .utils import get_user_agent, get_client_metadata
from .config import (
    CLIENT_ID,
    CLIENT_SECRET,
    SCOPES,
    CREDENTIAL_FILE,
    CODE_ASSIST_ENDPOINT,
    GEMINI_AUTH_PASSWORD,
    TOKEN_URI,
    AUTH_URI,
    OAUTH_REDIRECT_URI,
    OAUTH_CALLBACK_PORT,
    ONBOARD_POLL_INTERVAL,
    ONBOARD_MAX_RETRIES,
    ISO_DATE_FORMAT,
)

logger = logging.getLogger(__name__)

# --- Global State ---
_credentials: Optional[Credentials] = None
_user_project_id: Optional[str] = None
_onboarding_complete: bool = False
_credentials_from_env: bool = False


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    auth_code: Optional[str] = None

    def do_GET(self) -> None:
        query_components = parse_qs(urlparse(self.path).query)
        code = query_components.get("code", [None])[0]
        if code:
            _OAuthCallbackHandler.auth_code = code
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>OAuth authentication successful!</h1>"
                b"<p>You can close this window.</p>"
            )
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authentication failed.</h1><p>Please try again.</p>")

    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP server logging."""
        pass


def _parse_expiry(expiry_str: str) -> Optional[str]:
    """
    Parse and normalize expiry string to ISO format.

    Args:
        expiry_str: Expiry timestamp string in various formats

    Returns:
        Normalized expiry string or None if parsing fails
    """
    if not isinstance(expiry_str, str):
        return None

    try:
        if "+00:00" in expiry_str:
            parsed_expiry = datetime.fromisoformat(expiry_str)
        elif expiry_str.endswith("Z"):
            parsed_expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
        else:
            parsed_expiry = datetime.fromisoformat(expiry_str)

        timestamp = parsed_expiry.timestamp()
        return datetime.utcfromtimestamp(timestamp).strftime(ISO_DATE_FORMAT)
    except (ValueError, OSError) as e:
        logger.warning(f"Could not parse expiry format '{expiry_str}': {e}")
        return None


def _normalize_credentials_data(creds_data: dict) -> dict:
    """
    Normalize credential data to standard format.

    Args:
        creds_data: Raw credentials dictionary

    Returns:
        Normalized credentials dictionary
    """
    normalized = creds_data.copy()

    # Handle access_token -> token mapping
    if "access_token" in normalized and "token" not in normalized:
        normalized["token"] = normalized["access_token"]

    # Handle scope -> scopes mapping
    if "scope" in normalized and "scopes" not in normalized:
        normalized["scopes"] = normalized["scope"].split()

    # Handle expiry format
    if "expiry" in normalized:
        parsed_expiry = _parse_expiry(normalized["expiry"])
        if parsed_expiry:
            normalized["expiry"] = parsed_expiry
        else:
            del normalized["expiry"]

    return normalized


def _create_credentials_from_data(
    creds_data: dict, source: str = "unknown"
) -> Optional[Credentials]:
    """
    Create Credentials object from data dictionary.

    Args:
        creds_data: Credentials data dictionary
        source: Source identifier for logging

    Returns:
        Credentials object or None if creation fails
    """
    global _user_project_id

    try:
        normalized = _normalize_credentials_data(creds_data)
        credentials = Credentials.from_authorized_user_info(normalized, SCOPES)

        # Extract project_id if available
        if "project_id" in creds_data:
            _user_project_id = creds_data["project_id"]
            logger.info(f"Extracted project_id from {source}: {_user_project_id}")

        return credentials
    except Exception as e:
        logger.warning(f"Failed to create credentials from {source}: {e}")
        return None


def _create_minimal_credentials(creds_data: dict) -> Optional[Credentials]:
    """
    Create minimal credentials with just refresh token.

    Args:
        creds_data: Raw credentials data with refresh_token

    Returns:
        Credentials object or None if creation fails
    """
    try:
        minimal_data = {
            "client_id": creds_data.get("client_id", CLIENT_ID),
            "client_secret": creds_data.get("client_secret", CLIENT_SECRET),
            "refresh_token": creds_data["refresh_token"],
            "token_uri": TOKEN_URI,
        }
        return Credentials.from_authorized_user_info(minimal_data, SCOPES)
    except Exception as e:
        logger.error(f"Failed to create minimal credentials: {e}")
        return None


def _refresh_credentials(creds: Credentials) -> bool:
    """
    Attempt to refresh credentials.

    Args:
        creds: Credentials object to refresh

    Returns:
        True if refresh succeeded, False otherwise
    """
    if not creds.refresh_token:
        logger.warning("No refresh token available")
        return False

    try:
        creds.refresh(GoogleAuthRequest())
        logger.info("Credentials refreshed successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to refresh credentials: {e}")
        return False


def authenticate_user(request: Request) -> str:
    """
    Authenticate the user with multiple methods.

    Args:
        request: FastAPI request object

    Returns:
        Username/identifier of authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    # Check API key in query parameters
    api_key = request.query_params.get("key")
    if api_key and api_key == GEMINI_AUTH_PASSWORD:
        return "api_key_user"

    # Check x-goog-api-key header
    goog_api_key = request.headers.get("x-goog-api-key", "")
    if goog_api_key and goog_api_key == GEMINI_AUTH_PASSWORD:
        return "goog_api_key_user"

    # Check Bearer token
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:]
        if bearer_token == GEMINI_AUTH_PASSWORD:
            return "bearer_user"

    # Check Basic auth
    if auth_header.startswith("Basic "):
        try:
            encoded = auth_header[6:]
            decoded = base64.b64decode(encoded).decode("utf-8", "ignore")
            _, password = decoded.split(":", 1)
            if password == GEMINI_AUTH_PASSWORD:
                return "basic_user"
        except (ValueError, UnicodeDecodeError) as e:
            logger.debug(f"Basic auth decode failed: {e}")

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Basic"},
    )


def save_credentials(creds: Credentials, project_id: Optional[str] = None) -> None:
    """
    Save credentials to file.

    Args:
        creds: Credentials object to save
        project_id: Optional project ID to save
    """
    global _credentials_from_env

    if _credentials_from_env:
        # Only update project_id in existing file if needed
        if project_id and os.path.exists(CREDENTIAL_FILE):
            try:
                with open(CREDENTIAL_FILE, "r") as f:
                    existing_data = json.load(f)
                if "project_id" not in existing_data:
                    existing_data["project_id"] = project_id
                    with open(CREDENTIAL_FILE, "w") as f:
                        json.dump(existing_data, f, indent=2)
                    logger.info(f"Added project_id {project_id} to credential file")
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Could not update project_id in credential file: {e}")
        return

    creds_data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "token_uri": TOKEN_URI,
    }

    if creds.expiry:
        expiry_utc = creds.expiry
        if expiry_utc.tzinfo is None:
            expiry_utc = expiry_utc.replace(tzinfo=timezone.utc)
        creds_data["expiry"] = expiry_utc.isoformat()

    # Preserve or set project_id
    if project_id:
        creds_data["project_id"] = project_id
    elif os.path.exists(CREDENTIAL_FILE):
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                existing_data = json.load(f)
                if "project_id" in existing_data:
                    creds_data["project_id"] = existing_data["project_id"]
        except (IOError, json.JSONDecodeError):
            pass

    with open(CREDENTIAL_FILE, "w") as f:
        json.dump(creds_data, f, indent=2)


def _load_credentials_from_env() -> Optional[Credentials]:
    """Load credentials from GEMINI_CREDENTIALS environment variable."""
    global _credentials_from_env

    env_creds_json = os.getenv("GEMINI_CREDENTIALS")
    if not env_creds_json:
        return None

    try:
        raw_data = json.loads(env_creds_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GEMINI_CREDENTIALS JSON: {e}")
        return None

    if not raw_data.get("refresh_token"):
        logger.warning("No refresh_token in environment credentials")
        return None

    logger.info("Loading credentials from environment variable")

    # Try normal parsing first
    creds = _create_credentials_from_data(raw_data, "environment")
    if creds:
        _credentials_from_env = True
        if creds.expired:
            _refresh_credentials(creds)
        return creds

    # Try minimal credentials as fallback
    creds = _create_minimal_credentials(raw_data)
    if creds:
        _credentials_from_env = True
        _refresh_credentials(creds)
        return creds

    return None


def _load_credentials_from_file() -> Optional[Credentials]:
    """Load credentials from credential file."""
    global _credentials_from_env

    if not os.path.exists(CREDENTIAL_FILE):
        return None

    try:
        with open(CREDENTIAL_FILE, "r") as f:
            raw_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read credentials file: {e}")
        return None

    if not raw_data.get("refresh_token"):
        logger.warning("No refresh_token in credentials file")
        return None

    logger.info("Loading credentials from file")

    # Try normal parsing first
    creds = _create_credentials_from_data(raw_data, "file")
    if creds:
        _credentials_from_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        if creds.expired and _refresh_credentials(creds):
            save_credentials(creds)
        return creds

    # Try minimal credentials as fallback
    creds = _create_minimal_credentials(raw_data)
    if creds:
        _credentials_from_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        if _refresh_credentials(creds):
            save_credentials(creds)
        return creds

    return None


def _run_oauth_flow() -> Optional[Credentials]:
    """Run interactive OAuth flow."""
    global _credentials_from_env

    client_config = {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
        }
    }

    flow = Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=OAUTH_REDIRECT_URI
    )
    flow.oauth2session.scope = SCOPES

    auth_url, _ = flow.authorization_url(
        access_type="offline", prompt="consent", include_granted_scopes="true"
    )

    print(f"\n{'=' * 60}")
    print("AUTHENTICATION REQUIRED")
    print(f"{'=' * 60}")
    print(f"Please open this URL in your browser:\n{auth_url}")
    print(f"{'=' * 60}\n")
    logger.info(f"OAuth URL: {auth_url}")

    server = HTTPServer(("", OAUTH_CALLBACK_PORT), _OAuthCallbackHandler)
    server.handle_request()

    auth_code = _OAuthCallbackHandler.auth_code
    if not auth_code:
        return None

    # Patch oauthlib to ignore scope warnings
    import oauthlib.oauth2.rfc6749.parameters as oauth_params

    original_validate = oauth_params.validate_token_parameters
    oauth_params.validate_token_parameters = lambda p: None

    try:
        flow.fetch_token(code=auth_code)
        creds = flow.credentials
        _credentials_from_env = False
        save_credentials(creds)
        logger.info("Authentication successful!")
        return creds
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return None
    finally:
        oauth_params.validate_token_parameters = original_validate


def get_credentials(allow_oauth_flow: bool = True) -> Optional[Credentials]:
    """
    Load or obtain OAuth credentials.

    Args:
        allow_oauth_flow: Whether to run interactive OAuth if needed

    Returns:
        Credentials object or None if unavailable
    """
    global _credentials

    if _credentials and _credentials.token:
        return _credentials

    # Try environment variable first
    _credentials = _load_credentials_from_env()
    if _credentials:
        return _credentials

    # Try credentials file
    _credentials = _load_credentials_from_file()
    if _credentials:
        return _credentials

    # Run OAuth flow if allowed
    if allow_oauth_flow:
        _credentials = _run_oauth_flow()
        return _credentials

    logger.info("OAuth flow not allowed, returning None")
    return None


def onboard_user(creds: Credentials, project_id: str) -> None:
    """
    Ensure user is onboarded to Code Assist.

    Args:
        creds: Valid credentials
        project_id: Google Cloud project ID

    Raises:
        Exception: If onboarding fails
    """
    global _onboarding_complete

    if _onboarding_complete:
        return

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            save_credentials(creds)
        except Exception as e:
            raise Exception(f"Failed to refresh credentials: {e}")

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    payload = {
        "cloudaicompanionProject": project_id,
        "metadata": get_client_metadata(project_id),
    }

    try:
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        load_data = resp.json()

        # Get tier info
        tier = load_data.get("currentTier")
        if not tier:
            for allowed_tier in load_data.get("allowedTiers", []):
                if allowed_tier.get("isDefault"):
                    tier = allowed_tier
                    break
            if not tier:
                tier = {"id": "legacy-tier", "userDefinedCloudaicompanionProject": True}

        if tier.get("userDefinedCloudaicompanionProject") and not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable required.")

        if load_data.get("currentTier"):
            _onboarding_complete = True
            return

        # Run onboarding
        onboard_payload = {
            "tierId": tier.get("id"),
            "cloudaicompanionProject": project_id,
            "metadata": get_client_metadata(project_id),
        }

        for _ in range(ONBOARD_MAX_RETRIES):
            onboard_resp = requests.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                json=onboard_payload,
                headers=headers,
            )
            onboard_resp.raise_for_status()
            lro_data = onboard_resp.json()

            if lro_data.get("done"):
                _onboarding_complete = True
                return

            time.sleep(ONBOARD_POLL_INTERVAL)

        raise Exception("Onboarding timed out")

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        raise Exception(f"Onboarding failed: {error_text}")


def get_user_project_id(creds: Credentials) -> str:
    """
    Get the user's Google Cloud project ID.

    Args:
        creds: Valid credentials

    Returns:
        Project ID string

    Raises:
        Exception: If project ID cannot be determined
    """
    global _user_project_id

    # Priority 1: Environment variable
    env_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_project_id:
        logger.info(f"Using project ID from env: {env_project_id}")
        _user_project_id = env_project_id
        save_credentials(creds, _user_project_id)
        return _user_project_id

    # Priority 2: Cached value
    if _user_project_id:
        logger.info(f"Using cached project ID: {_user_project_id}")
        return _user_project_id

    # Priority 3: Credential file
    if os.path.exists(CREDENTIAL_FILE):
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                creds_data = json.load(f)
                cached = creds_data.get("project_id")
                if cached:
                    logger.info(f"Using project ID from file: {cached}")
                    _user_project_id = cached
                    return _user_project_id
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read project_id from file: {e}")

    # Priority 4: API discovery
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            save_credentials(creds)
        except Exception as e:
            logger.error(f"Failed to refresh credentials: {e}")

    if not creds.token:
        raise Exception("No valid access token for project ID discovery")

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    try:
        logger.info("Discovering project ID via API...")
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            json={"metadata": get_client_metadata()},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        discovered = data.get("cloudaicompanionProject")
        if not discovered:
            raise ValueError("Could not find project ID in API response")

        logger.info(f"Discovered project ID: {discovered}")
        _user_project_id = discovered
        save_credentials(creds, _user_project_id)
        return _user_project_id

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        raise Exception(f"Failed to discover project ID: {error_text}")
