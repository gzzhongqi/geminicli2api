"""
CLI tool for managing Gemini credentials.
Supports adding, listing, removing, and exporting credentials.

Usage:
    uv run python -m src.cli auth add [--name NAME]
    uv run python -m src.cli auth list
    uv run python -m src.cli auth remove <name>
    uv run python -m src.cli auth export [-o FILE] [--docker]
"""

import argparse
import json
import os
import sys
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from .config import (
    APP_NAME,
    AUTH_URI,
    CLIENT_ID,
    CLIENT_SECRET,
    OAUTH_CALLBACK_PORT,
    OAUTH_REDIRECT_URI,
    SCOPES,
    TOKEN_URI,
)


# --- Constants ---
CREDENTIALS_DIR = Path.home() / f".{APP_NAME}" / "credentials"


# --- Helper Classes ---
class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    auth_code: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        code = query.get("code", [None])[0]
        error = query.get("error", [None])[0]

        if error:
            _OAuthCallbackHandler.error = error
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                f"<h1>Authentication Failed</h1><p>Error: {error}</p>".encode()
            )
        elif code:
            _OAuthCallbackHandler.auth_code = code
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><head><style>"
                b"body { font-family: system-ui, sans-serif; display: flex; "
                b"justify-content: center; align-items: center; height: 100vh; "
                b"margin: 0; background: #f5f5f5; }"
                b".card { background: white; padding: 2rem 3rem; border-radius: 12px; "
                b"box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }"
                b"h1 { color: #1a73e8; margin-bottom: 0.5rem; }"
                b"p { color: #666; }"
                b"</style></head><body>"
                b"<div class='card'>"
                b"<h1>Authentication Successful!</h1>"
                b"<p>You can close this window and return to the terminal.</p>"
                b"</div></body></html>"
            )
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Invalid Request</h1>")

    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP server logging."""
        pass


# --- Credential Storage Functions ---
def _ensure_credentials_dir() -> Path:
    """Ensure credentials directory exists."""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    return CREDENTIALS_DIR


def _get_credential_path(name: str) -> Path:
    """Get path for a credential file."""
    return CREDENTIALS_DIR / f"{name}.json"


def _list_credentials() -> list[dict]:
    """List all stored credentials with metadata."""
    credentials = []
    if not CREDENTIALS_DIR.exists():
        return credentials

    for file in sorted(CREDENTIALS_DIR.glob("*.json")):
        try:
            with open(file, "r") as f:
                data = json.load(f)
            credentials.append(
                {
                    "name": file.stem,
                    "path": str(file),
                    "email": data.get("email", "unknown"),
                    "project_id": data.get("project_id", "unknown"),
                    "created": data.get("created_at", "unknown"),
                }
            )
        except (IOError, json.JSONDecodeError):
            credentials.append(
                {
                    "name": file.stem,
                    "path": str(file),
                    "email": "error reading file",
                    "project_id": "error",
                    "created": "error",
                }
            )
    return credentials


def _generate_credential_name() -> str:
    """Generate next available credential name."""
    existing = _list_credentials()
    existing_names = {c["name"] for c in existing}

    # Find next available number
    i = 1
    while f"credential_{i}" in existing_names:
        i += 1
    return f"credential_{i}"


def _get_user_email(creds: Credentials) -> Optional[str]:
    """Get user email from credentials using userinfo API."""
    import requests

    try:
        resp = requests.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {creds.token}"},
        )
        resp.raise_for_status()
        return resp.json().get("email")
    except Exception:
        return None


def _save_credential(
    creds: Credentials, name: str, email: Optional[str] = None
) -> Path:
    """Save credential to file."""
    _ensure_credentials_dir()

    creds_data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "token_uri": TOKEN_URI,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if email:
        creds_data["email"] = email

    if creds.expiry:
        expiry_utc = creds.expiry
        if expiry_utc.tzinfo is None:
            expiry_utc = expiry_utc.replace(tzinfo=timezone.utc)
        creds_data["expiry"] = expiry_utc.isoformat()

    path = _get_credential_path(name)
    with open(path, "w") as f:
        json.dump(creds_data, f, indent=2)

    return path


def _remove_credential(name: str) -> bool:
    """Remove a credential file."""
    path = _get_credential_path(name)
    if path.exists():
        path.unlink()
        return True
    return False


# --- OAuth Flow ---
def _run_oauth_flow() -> Optional[Credentials]:
    """Run standalone OAuth flow to obtain credentials."""
    # Reset class state
    _OAuthCallbackHandler.auth_code = None
    _OAuthCallbackHandler.error = None

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

    auth_url, _ = flow.authorization_url(
        access_type="offline", prompt="consent", include_granted_scopes="true"
    )

    print(f"\n{'=' * 60}")
    print("GOOGLE AUTHENTICATION")
    print(f"{'=' * 60}")
    print("\nOpening browser for authentication...")
    print(f"\nIf browser doesn't open, visit this URL:\n{auth_url}")
    print(f"\n{'=' * 60}\n")

    # Try to open browser
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass  # Browser open is optional

    # Start local server to receive callback
    try:
        server = HTTPServer(("", OAUTH_CALLBACK_PORT), _OAuthCallbackHandler)
        server.timeout = 300  # 5 minute timeout
        server.handle_request()
    except OSError as e:
        print(f"Error: Could not start callback server on port {OAUTH_CALLBACK_PORT}")
        print(f"Details: {e}")
        return None

    if _OAuthCallbackHandler.error:
        print(f"Authentication error: {_OAuthCallbackHandler.error}")
        return None

    auth_code = _OAuthCallbackHandler.auth_code
    if not auth_code:
        print("Authentication cancelled or timed out.")
        return None

    # Patch oauthlib to ignore scope and token validation warnings
    import oauthlib.oauth2.rfc6749.parameters as oauth_params
    from oauthlib.oauth2.rfc6749 import tokens as oauth_tokens

    original_validate_params = oauth_params.validate_token_parameters
    original_parse_token = getattr(oauth_params, "parse_token_response", None)
    original_parse = None  # Initialize for potential later use

    oauth_params.validate_token_parameters = lambda p: None

    # Some versions use different validation
    if hasattr(oauth_tokens, "parse_token"):
        original_parse = oauth_tokens.parse_token  # type: ignore[attr-defined]
        oauth_tokens.parse_token = lambda response, scope: response  # type: ignore[attr-defined]

    try:
        flow.fetch_token(code=auth_code)
        creds = flow.credentials
        # flow.credentials returns OAuth2 Credentials for installed app flow
        if not isinstance(creds, Credentials):
            return None
        return creds
    except KeyError as e:
        # Handle missing 'access_token' in response - try manual token fetch
        print(f"Token exchange issue: {e}, attempting manual fetch...")
        try:
            import requests

            token_response = requests.post(
                TOKEN_URI,
                data={
                    "code": auth_code,
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "redirect_uri": OAUTH_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
            token_data = token_response.json()
            if "error" in token_data:
                print(
                    f"Token error: {token_data.get('error_description', token_data['error'])}"
                )
                return None

            # Create credentials manually
            creds = Credentials(
                token=token_data.get("access_token"),
                refresh_token=token_data.get("refresh_token"),
                token_uri=TOKEN_URI,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                scopes=SCOPES,
            )
            return creds
        except Exception as e2:
            print(f"Manual token fetch failed: {e2}")
            return None
    except Exception as e:
        print(f"Failed to exchange code for token: {e}")
        return None
    finally:
        oauth_params.validate_token_parameters = original_validate_params
        if hasattr(oauth_tokens, "parse_token"):
            oauth_tokens.parse_token = original_parse  # type: ignore[attr-defined]


# --- CLI Commands ---
def cmd_auth_add(args: argparse.Namespace) -> int:
    """Add a new credential via OAuth flow."""
    print("Starting OAuth authentication...")

    creds = _run_oauth_flow()
    if not creds:
        print("Authentication failed.")
        return 1

    # Refresh to ensure we have a valid token
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
        except Exception as e:
            print(f"Warning: Could not refresh token: {e}")

    # Get email
    email = _get_user_email(creds)
    if email:
        print(f"Authenticated as: {email}")

    # Determine name
    name = args.name if args.name else _generate_credential_name()

    # Check if name already exists
    if _get_credential_path(name).exists():
        response = input(f"Credential '{name}' already exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            return 1

    # Save credential
    path = _save_credential(creds, name, email)

    print(f"\nCredential saved successfully!")
    print(f"  Name: {name}")
    print(f"  Path: {path}")
    if email:
        print(f"  Email: {email}")

    return 0


def cmd_auth_list(args: argparse.Namespace) -> int:
    """List all stored credentials."""
    credentials = _list_credentials()

    if not credentials:
        print("No credentials found.")
        print(f"\nCredentials directory: {CREDENTIALS_DIR}")
        print("Run 'uv run python -m src.cli auth add' to add a credential.")
        return 0

    print(f"Found {len(credentials)} credential(s):\n")
    print(f"{'Name':<20} {'Email':<35} {'Created':<25}")
    print("-" * 80)

    for cred in credentials:
        created = cred["created"]
        if created != "unknown" and created != "error":
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                created = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                pass

        email = cred["email"]
        if len(email) > 33:
            email = email[:30] + "..."

        print(f"{cred['name']:<20} {email:<35} {created:<25}")

    print(f"\nCredentials directory: {CREDENTIALS_DIR}")
    return 0


def cmd_auth_remove(args: argparse.Namespace) -> int:
    """Remove a credential."""
    name = args.name

    if not _get_credential_path(name).exists():
        print(f"Credential '{name}' not found.")
        return 1

    response = input(f"Remove credential '{name}'? [y/N]: ")
    if response.lower() != "y":
        print("Cancelled.")
        return 0

    if _remove_credential(name):
        print(f"Credential '{name}' removed.")
        return 0
    else:
        print(f"Failed to remove credential '{name}'.")
        return 1


def cmd_auth_export(args: argparse.Namespace) -> int:
    """Export credentials as environment variables."""
    credentials = _list_credentials()

    if not credentials:
        print("No credentials found to export.")
        return 1

    lines = []

    if args.docker:
        # Docker-compose format
        lines.append("# Add to your docker-compose.yml environment section:")
        lines.append("environment:")
        for i, cred in enumerate(credentials, 1):
            path = Path(cred["path"])
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                # Remove non-essential fields for export
                export_data = {
                    "client_id": data.get("client_id", CLIENT_ID),
                    "client_secret": data.get("client_secret", CLIENT_SECRET),
                    "refresh_token": data["refresh_token"],
                    "token_uri": data.get("token_uri", TOKEN_URI),
                }
                json_str = json.dumps(export_data, separators=(",", ":"))
                lines.append(f"  - GEMINI_CREDENTIALS_{i}='{json_str}'")
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read credential '{cred['name']}': {e}")
                continue
    else:
        # .env format
        lines.append("# Gemini Credentials - Add to your .env file")
        lines.append(f"# Exported at {datetime.now(timezone.utc).isoformat()}")
        lines.append("")
        for i, cred in enumerate(credentials, 1):
            path = Path(cred["path"])
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                # Remove non-essential fields for export
                export_data = {
                    "client_id": data.get("client_id", CLIENT_ID),
                    "client_secret": data.get("client_secret", CLIENT_SECRET),
                    "refresh_token": data["refresh_token"],
                    "token_uri": data.get("token_uri", TOKEN_URI),
                }
                json_str = json.dumps(export_data, separators=(",", ":"))
                lines.append(f"# {cred['name']} ({cred['email']})")
                lines.append(f"GEMINI_CREDENTIALS_{i}='{json_str}'")
                lines.append("")
            except (IOError, json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read credential '{cred['name']}': {e}")
                continue

    output = "\n".join(lines)

    if args.output:
        # Write to file
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Exported {len(credentials)} credential(s) to {args.output}")
    else:
        # Print to stdout
        print(output)

    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Manage Gemini credentials for geminicli2api",
        prog="python -m src.cli",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # auth command group
    auth_parser = subparsers.add_parser("auth", help="Manage authentication")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command")

    # auth add
    add_parser = auth_subparsers.add_parser("add", help="Add a new credential")
    add_parser.add_argument(
        "--name", "-n", type=str, help="Custom name for the credential"
    )

    # auth list
    auth_subparsers.add_parser("list", help="List all credentials")

    # auth remove
    remove_parser = auth_subparsers.add_parser("remove", help="Remove a credential")
    remove_parser.add_argument(
        "name", type=str, help="Name of the credential to remove"
    )

    # auth export
    export_parser = auth_subparsers.add_parser(
        "export", help="Export credentials as environment variables"
    )
    export_parser.add_argument(
        "--output", "-o", type=str, help="Output file (default: stdout)"
    )
    export_parser.add_argument(
        "--docker", action="store_true", help="Export in docker-compose format"
    )

    args = parser.parse_args()

    if args.command == "auth":
        if args.auth_command == "add":
            return cmd_auth_add(args)
        elif args.auth_command == "list":
            return cmd_auth_list(args)
        elif args.auth_command == "remove":
            return cmd_auth_remove(args)
        elif args.auth_command == "export":
            return cmd_auth_export(args)
        else:
            auth_parser.print_help()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
