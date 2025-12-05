"""
Authentication module for Geminicli2api.
Handles OAuth2 authentication with Google APIs.
Supports multiple credentials with round-robin selection and automatic fallback.
"""

import os
import json
import base64
import time
import logging
import threading
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, List, Tuple
from urllib.parse import urlparse, parse_qs

from fastapi import Request, HTTPException
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleAuthRequest
import requests

from ..utils import get_user_agent, get_client_metadata
from ..config import (
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

# --- Constants ---
CREDENTIAL_RECOVERY_TIME = 300  # 5 minutes before retrying failed credential

# --- Global State ---
_credentials: Optional[Credentials] = None
_user_project_id: Optional[str] = None
_onboarding_complete: bool = False
_credentials_from_env: bool = False


# --- Credential Pool for Multiple Credentials ---
@dataclass
class CredentialEntry:
    """Represents a single credential with its metadata."""

    credentials: Credentials
    project_id: Optional[str] = None
    source: str = "unknown"
    index: int = 0
    failed_at: Optional[float] = None
    onboarding_complete: bool = False
    from_env: bool = False


class CredentialPool:
    """
    Manages multiple Google OAuth credentials with round-robin selection
    and automatic fallback on failure.
    """

    def __init__(self):
        self._entries: List[CredentialEntry] = []
        self._current_index: int = 0
        self._lock = threading.Lock()

    def add(self, entry: CredentialEntry) -> None:
        """Add a credential entry to the pool."""
        with self._lock:
            entry.index = len(self._entries)
            self._entries.append(entry)
            logger.info(f"Added credential #{entry.index + 1} from {entry.source}")

    def size(self) -> int:
        """Return the number of credentials in the pool."""
        return len(self._entries)

    def is_empty(self) -> bool:
        """Check if the pool has no credentials."""
        return len(self._entries) == 0

    def get_next(self) -> Optional[Tuple[CredentialEntry, int]]:
        """
        Get the next available credential using round-robin selection.
        Skips credentials that are temporarily marked as failed.

        Returns:
            Tuple of (CredentialEntry, index) or None if no credentials available
        """
        with self._lock:
            if not self._entries:
                return None

            current_time = time.time()
            attempts = 0
            total = len(self._entries)

            while attempts < total:
                entry = self._entries[self._current_index]
                idx = self._current_index
                self._current_index = (self._current_index + 1) % total

                # Check if credential has recovered from failure
                if entry.failed_at is not None:
                    if current_time - entry.failed_at >= CREDENTIAL_RECOVERY_TIME:
                        entry.failed_at = None
                        logger.info(f"Credential #{idx + 1} recovered, now available")
                    else:
                        attempts += 1
                        continue

                logger.debug(f"Selected credential #{idx + 1} (round-robin)")
                return (entry, idx)

            # All credentials failed, try the least recently failed one
            logger.warning("All credentials marked as failed, trying oldest failure")
            oldest_entry = min(self._entries, key=lambda e: e.failed_at or 0)
            oldest_entry.failed_at = None
            return (oldest_entry, oldest_entry.index)

    def get_fallback(self, exclude_index: int) -> Optional[Tuple[CredentialEntry, int]]:
        """
        Get a fallback credential, excluding the specified index.

        Args:
            exclude_index: Index of credential to skip

        Returns:
            Tuple of (CredentialEntry, index) or None if no fallback available
        """
        with self._lock:
            if len(self._entries) <= 1:
                return None

            current_time = time.time()

            for i, entry in enumerate(self._entries):
                if i == exclude_index:
                    continue

                # Check if recovered or never failed
                if entry.failed_at is None:
                    logger.info(f"Using fallback credential #{i + 1}")
                    return (entry, i)
                elif current_time - entry.failed_at >= CREDENTIAL_RECOVERY_TIME:
                    entry.failed_at = None
                    logger.info(f"Using recovered fallback credential #{i + 1}")
                    return (entry, i)

            return None

    def mark_failed(self, index: int) -> None:
        """
        Mark a credential as temporarily failed.

        Args:
            index: Index of the failed credential
        """
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].failed_at = time.time()
                logger.warning(
                    f"Credential #{index + 1} marked as failed, "
                    f"will retry in {CREDENTIAL_RECOVERY_TIME}s"
                )

    def mark_success(self, index: int) -> None:
        """
        Mark a credential as successful (clear failed status).

        Args:
            index: Index of the successful credential
        """
        with self._lock:
            if 0 <= index < len(self._entries):
                if self._entries[index].failed_at is not None:
                    self._entries[index].failed_at = None
                    logger.info(f"Credential #{index + 1} marked as recovered")

    def get_entry(self, index: int) -> Optional[CredentialEntry]:
        """Get a specific credential entry by index."""
        with self._lock:
            if 0 <= index < len(self._entries):
                return self._entries[index]
            return None

    def update_project_id(self, index: int, project_id: str) -> None:
        """Update the project ID for a credential entry."""
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].project_id = project_id

    def set_onboarding_complete(self, index: int) -> None:
        """Mark onboarding as complete for a credential entry."""
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].onboarding_complete = True

    def get_stats(self) -> dict:
        """Get pool statistics for debugging."""
        with self._lock:
            current_time = time.time()
            return {
                "total": len(self._entries),
                "available": sum(
                    1
                    for e in self._entries
                    if e.failed_at is None
                    or current_time - e.failed_at >= CREDENTIAL_RECOVERY_TIME
                ),
                "failed": sum(
                    1
                    for e in self._entries
                    if e.failed_at is not None
                    and current_time - e.failed_at < CREDENTIAL_RECOVERY_TIME
                ),
                "current_index": self._current_index,
            }


# Global credential pool instance
_credential_pool: Optional[CredentialPool] = None


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
    # If no password is set, skip authentication
    if not GEMINI_AUTH_PASSWORD:
        return "anonymous"

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


# --- Multi-Credential Loading Functions ---


def _load_credential_entry_from_json(
    json_str: str, source: str
) -> Optional[CredentialEntry]:
    """
    Create a CredentialEntry from JSON string.

    Args:
        json_str: JSON string containing credentials
        source: Source identifier for logging

    Returns:
        CredentialEntry or None if parsing fails
    """
    try:
        raw_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse credentials JSON from {source}: {e}")
        return None

    if not raw_data.get("refresh_token"):
        logger.warning(f"No refresh_token in credentials from {source}")
        return None

    # Try normal parsing first
    creds = _create_credentials_from_data(raw_data, source)
    if not creds:
        # Try minimal credentials as fallback
        creds = _create_minimal_credentials(raw_data)

    if not creds:
        return None

    if creds.expired:
        _refresh_credentials(creds)

    project_id = raw_data.get("project_id")

    return CredentialEntry(
        credentials=creds,
        project_id=project_id,
        source=source,
        from_env=True,
    )


def _load_credential_entry_from_file(
    file_path: str, source: str
) -> Optional[CredentialEntry]:
    """
    Create a CredentialEntry from a file.

    Args:
        file_path: Path to the credentials file
        source: Source identifier for logging

    Returns:
        CredentialEntry or None if loading fails
    """
    if not os.path.exists(file_path):
        logger.warning(f"Credentials file not found: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            raw_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read credentials file {file_path}: {e}")
        return None

    if not raw_data.get("refresh_token"):
        logger.warning(f"No refresh_token in credentials file {file_path}")
        return None

    # Try normal parsing first
    creds = _create_credentials_from_data(raw_data, source)
    if not creds:
        # Try minimal credentials as fallback
        creds = _create_minimal_credentials(raw_data)

    if not creds:
        return None

    if creds.expired:
        _refresh_credentials(creds)

    project_id = raw_data.get("project_id")

    return CredentialEntry(
        credentials=creds,
        project_id=project_id,
        source=source,
        from_env=False,
    )


def _load_multiple_credentials_from_env() -> List[CredentialEntry]:
    """
    Load multiple credentials from environment variables.
    Supports patterns: GEMINI_CREDENTIALS_1, GEMINI_CREDENTIALS_2, etc.

    Returns:
        List of CredentialEntry objects
    """
    entries: List[CredentialEntry] = []

    # Find all GEMINI_CREDENTIALS_N environment variables
    pattern = re.compile(r"^GEMINI_CREDENTIALS_(\d+)$")
    indexed_vars: List[Tuple[int, str, str]] = []

    for key, value in os.environ.items():
        match = pattern.match(key)
        if match:
            index = int(match.group(1))
            indexed_vars.append((index, key, value))

    # Sort by index to maintain order
    indexed_vars.sort(key=lambda x: x[0])

    for index, key, value in indexed_vars:
        entry = _load_credential_entry_from_json(value, f"env:{key}")
        if entry:
            entries.append(entry)

    if entries:
        logger.info(f"Loaded {len(entries)} credentials from indexed env vars")

    return entries


def _load_multiple_credentials_from_files() -> List[CredentialEntry]:
    """
    Load multiple credentials from files specified in GEMINI_CREDENTIAL_FILES.
    Format: comma-separated list of file paths.

    Returns:
        List of CredentialEntry objects
    """
    entries: List[CredentialEntry] = []

    files_str = os.getenv("GEMINI_CREDENTIAL_FILES", "")
    if not files_str:
        return entries

    file_paths = [f.strip() for f in files_str.split(",") if f.strip()]

    for i, file_path in enumerate(file_paths):
        # Handle relative paths
        if not os.path.isabs(file_path):
            from ..config import SCRIPT_DIR

            file_path = os.path.join(SCRIPT_DIR, file_path)

        entry = _load_credential_entry_from_file(file_path, f"file:{file_path}")
        if entry:
            entries.append(entry)

    if entries:
        logger.info(f"Loaded {len(entries)} credentials from files")

    return entries


def initialize_credential_pool() -> CredentialPool:
    """
    Initialize the credential pool with all available credentials.
    Loads from multiple sources in order of priority.

    Returns:
        Initialized CredentialPool
    """
    global _credential_pool

    if _credential_pool is not None and not _credential_pool.is_empty():
        return _credential_pool

    _credential_pool = CredentialPool()

    # Priority 1: Multiple indexed environment variables (GEMINI_CREDENTIALS_N)
    env_entries = _load_multiple_credentials_from_env()
    for entry in env_entries:
        _credential_pool.add(entry)

    # Priority 2: Multiple files (GEMINI_CREDENTIAL_FILES)
    file_entries = _load_multiple_credentials_from_files()
    for entry in file_entries:
        _credential_pool.add(entry)

    # Priority 3: Single GEMINI_CREDENTIALS env var (backward compatibility)
    if _credential_pool.is_empty():
        single_creds_json = os.getenv("GEMINI_CREDENTIALS")
        if single_creds_json:
            entry = _load_credential_entry_from_json(
                single_creds_json, "env:GEMINI_CREDENTIALS"
            )
            if entry:
                _credential_pool.add(entry)

    # Priority 4: Default credential file (backward compatibility)
    if _credential_pool.is_empty():
        entry = _load_credential_entry_from_file(
            CREDENTIAL_FILE, f"file:{CREDENTIAL_FILE}"
        )
        if entry:
            _credential_pool.add(entry)

    stats = _credential_pool.get_stats()
    logger.info(
        f"Credential pool initialized: {stats['total']} total, "
        f"{stats['available']} available"
    )

    return _credential_pool


def get_credential_pool() -> CredentialPool:
    """
    Get or initialize the credential pool.

    Returns:
        The global CredentialPool instance
    """
    global _credential_pool

    if _credential_pool is None:
        return initialize_credential_pool()

    return _credential_pool


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
        # flow.credentials returns OAuth2 Credentials for installed app flow
        if not isinstance(creds, Credentials):
            logger.error("Unexpected credentials type from OAuth flow")
            return None
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
    Load or obtain OAuth credentials (backward compatible single-credential mode).

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


def get_next_credential(
    allow_oauth_flow: bool = True,
) -> Optional[Tuple[Credentials, int]]:
    """
    Get the next available credential from the pool using round-robin selection.

    Args:
        allow_oauth_flow: Whether to run interactive OAuth if pool is empty

    Returns:
        Tuple of (Credentials, credential_index) or None if unavailable
    """
    pool = get_credential_pool()

    if pool.is_empty():
        # Fallback to single credential mode / OAuth flow
        creds = get_credentials(allow_oauth_flow)
        if creds:
            return (creds, -1)  # -1 indicates single credential mode
        return None

    result = pool.get_next()
    if result:
        entry, idx = result
        # Refresh if needed
        if entry.credentials.expired and entry.credentials.refresh_token:
            try:
                entry.credentials.refresh(GoogleAuthRequest())
                logger.debug(f"Refreshed credential #{idx + 1}")
            except Exception as e:
                logger.warning(f"Failed to refresh credential #{idx + 1}: {e}")
                pool.mark_failed(idx)
                # Try to get fallback
                fallback = pool.get_fallback(idx)
                if fallback:
                    return (fallback[0].credentials, fallback[1])
                return None

        return (entry.credentials, idx)

    return None


def get_fallback_credential(exclude_index: int) -> Optional[Tuple[Credentials, int]]:
    """
    Get a fallback credential when the current one fails.

    Args:
        exclude_index: Index of credential to skip

    Returns:
        Tuple of (Credentials, credential_index) or None if no fallback available
    """
    if exclude_index == -1:
        # Single credential mode, no fallback
        return None

    pool = get_credential_pool()
    result = pool.get_fallback(exclude_index)

    if result:
        entry, idx = result
        # Refresh if needed
        if entry.credentials.expired and entry.credentials.refresh_token:
            try:
                entry.credentials.refresh(GoogleAuthRequest())
            except Exception as e:
                logger.warning(f"Failed to refresh fallback credential #{idx + 1}: {e}")
                pool.mark_failed(idx)
                return None

        return (entry.credentials, idx)

    return None


def mark_credential_failed(index: int) -> None:
    """
    Mark a credential as failed.

    Args:
        index: Index of the failed credential (-1 for single credential mode)
    """
    if index == -1:
        return  # Single credential mode, nothing to mark

    pool = get_credential_pool()
    pool.mark_failed(index)


def mark_credential_success(index: int) -> None:
    """
    Mark a credential as successful (clear failed status).

    Args:
        index: Index of the successful credential
    """
    if index == -1:
        return  # Single credential mode

    pool = get_credential_pool()
    pool.mark_success(index)


def get_credential_project_id(index: int) -> Optional[str]:
    """
    Get the project ID associated with a credential.

    Args:
        index: Credential index

    Returns:
        Project ID or None
    """
    if index == -1:
        return None  # Use global project ID discovery

    pool = get_credential_pool()
    entry = pool.get_entry(index)
    return entry.project_id if entry else None


def set_credential_project_id(index: int, project_id: str) -> None:
    """
    Set the project ID for a credential.

    Args:
        index: Credential index
        project_id: Project ID to set
    """
    if index == -1:
        return  # Single credential mode uses global

    pool = get_credential_pool()
    pool.update_project_id(index, project_id)


def is_credential_onboarded(index: int) -> bool:
    """
    Check if a credential has completed onboarding.

    Args:
        index: Credential index

    Returns:
        True if onboarded
    """
    if index == -1:
        return _onboarding_complete

    pool = get_credential_pool()
    entry = pool.get_entry(index)
    return entry.onboarding_complete if entry else False


def set_credential_onboarded(index: int) -> None:
    """
    Mark a credential as having completed onboarding.

    Args:
        index: Credential index
    """
    global _onboarding_complete

    if index == -1:
        _onboarding_complete = True
        return

    pool = get_credential_pool()
    pool.set_onboarding_complete(index)


def get_pool_stats() -> dict:
    """
    Get credential pool statistics.

    Returns:
        Dictionary with pool statistics
    """
    pool = get_credential_pool()
    return pool.get_stats()


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


def discover_project_id_for_credential(creds: Credentials) -> Optional[str]:
    """
    Discover project ID for a specific credential via API.

    Unlike get_user_project_id(), this function does NOT use global cache
    and always queries the API. Used for multi-credential mode where each
    credential has its own project ID.

    Args:
        creds: Valid credentials

    Returns:
        Project ID or None if discovery fails
    """
    # Refresh credentials if needed
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
        except Exception as e:
            logger.error(f"Failed to refresh credentials: {e}")
            return None

    if not creds.token:
        logger.error("No valid access token for project ID discovery")
        return None

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    try:
        logger.info("Discovering project ID via API for credential...")
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            json={"metadata": get_client_metadata()},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        discovered = data.get("cloudaicompanionProject")

        # If no project ID, try auto-onboarding to free tier
        if not discovered:
            logger.info("No project ID in response, attempting auto-onboard...")
            discovered = _auto_onboard_to_free_tier(creds, headers)

        if discovered:
            logger.info(f"Discovered project ID: {discovered}")

        return discovered

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        logger.error(f"Failed to discover project ID: {error_text}")
        return None
    except Exception as e:
        logger.error(f"Error discovering project ID: {e}")
        return None


def _auto_onboard_to_free_tier(creds: Credentials, headers: dict) -> Optional[str]:
    """
    Automatically onboard user to free tier and return project ID.

    This is called when loadCodeAssist doesn't return a cloudaicompanionProject,
    which happens for accounts that haven't selected a tier yet.

    Args:
        creds: Valid credentials
        headers: HTTP headers for API calls

    Returns:
        Project ID if onboarding succeeds, None otherwise
    """
    logger.info("Auto-onboarding to free tier...")

    try:
        # Start onboarding to free-tier
        onboard_resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
            json={"tierId": "free-tier", "metadata": get_client_metadata()},
            headers=headers,
        )
        onboard_resp.raise_for_status()
        logger.debug(f"Onboarding initiated: {onboard_resp.json()}")

        # Poll loadCodeAssist until we get project ID
        for attempt in range(ONBOARD_MAX_RETRIES):
            time.sleep(ONBOARD_POLL_INTERVAL)

            resp = requests.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                json={"metadata": get_client_metadata()},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            project_id = data.get("cloudaicompanionProject")
            if project_id:
                logger.info(f"Auto-onboarding successful, project ID: {project_id}")
                return project_id

            # Check if we have currentTier but no project ID yet
            if data.get("currentTier"):
                logger.debug(f"Waiting for project ID (attempt {attempt + 1})...")
            else:
                logger.debug(
                    f"Waiting for onboarding to complete (attempt {attempt + 1})..."
                )

        logger.warning("Auto-onboarding timed out waiting for project ID")
        return None

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        logger.error(f"Auto-onboarding failed: {error_text}")
        return None
    except Exception as e:
        logger.error(f"Auto-onboarding error: {e}")
        return None


def get_user_project_id(creds: Credentials) -> str:
    """
    Get the user's Google Cloud project ID.

    If the account hasn't been onboarded to Gemini Code Assist yet,
    this will automatically onboard to the free tier.

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
                    return cached  # Return local variable which is narrowed to str
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

        # If no project ID, try auto-onboarding to free tier
        if not discovered:
            logger.info("No project ID in response, attempting auto-onboard...")
            discovered = _auto_onboard_to_free_tier(creds, headers)

        if not discovered:
            raise ValueError(
                "Could not get project ID. The account may need manual onboarding. "
                "Try visiting https://codeassist.google.com/ to set up your account."
            )

        logger.info(f"Discovered project ID: {discovered}")
        _user_project_id = discovered
        save_credentials(creds, _user_project_id)
        return discovered  # Return local variable which is narrowed to str

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        raise Exception(f"Failed to discover project ID: {error_text}")
