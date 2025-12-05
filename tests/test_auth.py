"""Tests for authentication module."""

import base64
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.services.auth import (
    CredentialPool,
    CredentialEntry,
    authenticate_user,
    CREDENTIAL_RECOVERY_TIME,
)


# --- Test authenticate_user ---


class TestAuthenticateUser:
    """Tests for authenticate_user function."""

    def _create_mock_request(
        self,
        query_params: dict = None,
        headers: dict = None,
    ) -> MagicMock:
        """Create a mock FastAPI request."""
        request = MagicMock()
        request.query_params = query_params or {}
        request.headers = headers or {}
        return request

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", None)
    def test_no_password_set_returns_anonymous(self):
        """When GEMINI_AUTH_PASSWORD is not set, should return anonymous."""
        request = self._create_mock_request()
        result = authenticate_user(request)
        assert result == "anonymous"

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "")
    def test_empty_password_returns_anonymous(self):
        """When GEMINI_AUTH_PASSWORD is empty, should return anonymous."""
        request = self._create_mock_request()
        result = authenticate_user(request)
        assert result == "anonymous"

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "secret123")
    def test_valid_api_key_in_query(self):
        """Valid API key in query params should authenticate."""
        request = self._create_mock_request(query_params={"key": "secret123"})
        result = authenticate_user(request)
        assert result == "api_key_user"

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "secret123")
    def test_valid_goog_api_key_header(self):
        """Valid x-goog-api-key header should authenticate."""
        request = self._create_mock_request(headers={"x-goog-api-key": "secret123"})
        result = authenticate_user(request)
        assert result == "goog_api_key_user"

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "secret123")
    def test_valid_bearer_token(self):
        """Valid Bearer token should authenticate."""
        request = self._create_mock_request(
            headers={"authorization": "Bearer secret123"}
        )
        result = authenticate_user(request)
        assert result == "bearer_user"

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "secret123")
    def test_valid_basic_auth(self):
        """Valid Basic auth should authenticate."""
        # Basic auth: base64("user:secret123")
        encoded = base64.b64encode(b"user:secret123").decode("utf-8")
        request = self._create_mock_request(
            headers={"authorization": f"Basic {encoded}"}
        )
        result = authenticate_user(request)
        assert result == "basic_user"

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "secret123")
    def test_invalid_api_key_raises_401(self):
        """Invalid API key should raise HTTPException 401."""
        request = self._create_mock_request(query_params={"key": "wrongkey"})
        with pytest.raises(HTTPException) as exc_info:
            authenticate_user(request)
        assert exc_info.value.status_code == 401

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "secret123")
    def test_invalid_bearer_token_raises_401(self):
        """Invalid Bearer token should raise HTTPException 401."""
        request = self._create_mock_request(
            headers={"authorization": "Bearer wrongtoken"}
        )
        with pytest.raises(HTTPException) as exc_info:
            authenticate_user(request)
        assert exc_info.value.status_code == 401

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "secret123")
    def test_no_credentials_raises_401(self):
        """Missing credentials should raise HTTPException 401."""
        request = self._create_mock_request()
        with pytest.raises(HTTPException) as exc_info:
            authenticate_user(request)
        assert exc_info.value.status_code == 401

    @patch("src.services.auth.GEMINI_AUTH_PASSWORD", "secret123")
    def test_malformed_basic_auth_raises_401(self):
        """Malformed Basic auth should raise HTTPException 401."""
        request = self._create_mock_request(
            headers={"authorization": "Basic notvalidbase64!!!"}
        )
        with pytest.raises(HTTPException) as exc_info:
            authenticate_user(request)
        assert exc_info.value.status_code == 401


# --- Test CredentialPool ---


class TestCredentialPool:
    """Tests for CredentialPool class."""

    def _create_mock_credential(self, token: str = "token") -> MagicMock:
        """Create a mock Credentials object."""
        creds = MagicMock()
        creds.token = token
        creds.expired = False
        creds.refresh_token = "refresh_token"
        return creds

    def _create_entry(
        self,
        token: str = "token",
        source: str = "test",
        project_id: str = None,
    ) -> CredentialEntry:
        """Create a CredentialEntry for testing."""
        return CredentialEntry(
            credentials=self._create_mock_credential(token),
            project_id=project_id,
            source=source,
        )

    def test_empty_pool(self):
        """Empty pool should return None on get_next."""
        pool = CredentialPool()
        assert pool.is_empty()
        assert pool.size() == 0
        assert pool.get_next() is None

    def test_add_credential(self):
        """Adding credentials should increase pool size."""
        pool = CredentialPool()
        entry = self._create_entry(token="token1")

        pool.add(entry)

        assert pool.size() == 1
        assert not pool.is_empty()

    def test_get_next_single_credential(self):
        """get_next with single credential should return it."""
        pool = CredentialPool()
        entry = self._create_entry(token="token1")
        pool.add(entry)

        result = pool.get_next()

        assert result is not None
        entry, idx = result
        assert entry.credentials.token == "token1"
        assert idx == 0

    def test_round_robin_selection(self):
        """get_next should rotate through credentials."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1", source="cred1"))
        pool.add(self._create_entry(token="token2", source="cred2"))
        pool.add(self._create_entry(token="token3", source="cred3"))

        # First round
        result1 = pool.get_next()
        result2 = pool.get_next()
        result3 = pool.get_next()

        assert result1[0].credentials.token == "token1"
        assert result2[0].credentials.token == "token2"
        assert result3[0].credentials.token == "token3"

        # Second round (should wrap around)
        result4 = pool.get_next()
        assert result4[0].credentials.token == "token1"

    def test_mark_failed_skips_credential(self):
        """Failed credentials should be skipped."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))
        pool.add(self._create_entry(token="token2"))

        # Mark first credential as failed
        pool.mark_failed(0)

        # Should skip to second credential
        result = pool.get_next()
        assert result[0].credentials.token == "token2"

    def test_mark_success_clears_failed(self):
        """mark_success should clear failed status."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))

        pool.mark_failed(0)
        pool.mark_success(0)

        result = pool.get_next()
        assert result is not None
        assert result[0].credentials.token == "token1"

    def test_get_fallback_excludes_index(self):
        """get_fallback should return credential excluding specified index."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))
        pool.add(self._create_entry(token="token2"))

        result = pool.get_fallback(exclude_index=0)

        assert result is not None
        assert result[0].credentials.token == "token2"
        assert result[1] == 1

    def test_get_fallback_single_credential_returns_none(self):
        """get_fallback with single credential should return None."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))

        result = pool.get_fallback(exclude_index=0)

        assert result is None

    def test_get_fallback_all_failed_returns_none(self):
        """get_fallback should return None if all other credentials failed."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))
        pool.add(self._create_entry(token="token2"))

        pool.mark_failed(1)  # Mark the fallback as failed

        result = pool.get_fallback(exclude_index=0)

        assert result is None

    @patch("src.services.auth.CREDENTIAL_RECOVERY_TIME", 1)
    def test_credential_recovery_after_timeout(self):
        """Failed credential should recover after CREDENTIAL_RECOVERY_TIME."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))
        pool.add(self._create_entry(token="token2"))

        pool.mark_failed(0)

        # Immediately after, should skip to token2
        result = pool.get_next()
        assert result[0].credentials.token == "token2"

        # Wait for recovery
        time.sleep(1.1)

        # Now token1 should be available again
        # Reset index to test recovery
        pool._current_index = 0
        result = pool.get_next()
        assert result[0].credentials.token == "token1"

    def test_all_credentials_failed_returns_oldest(self):
        """When all credentials failed, should return the oldest failure."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))
        pool.add(self._create_entry(token="token2"))

        # Mark all as failed with different times
        pool.mark_failed(0)
        time.sleep(0.01)  # Small delay
        pool.mark_failed(1)

        # Should return the oldest failure (token1)
        result = pool.get_next()
        assert result is not None
        assert result[0].credentials.token == "token1"

    def test_get_stats(self):
        """get_stats should return correct statistics."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))
        pool.add(self._create_entry(token="token2"))
        pool.add(self._create_entry(token="token3"))

        pool.mark_failed(1)

        stats = pool.get_stats()

        assert stats["total"] == 3
        assert stats["available"] == 2
        assert stats["failed"] == 1

    def test_update_project_id(self):
        """update_project_id should set project_id for credential."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))

        pool.update_project_id(0, "my-project")

        entry = pool.get_entry(0)
        assert entry.project_id == "my-project"

    def test_set_onboarding_complete(self):
        """set_onboarding_complete should mark credential as onboarded."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))

        pool.set_onboarding_complete(0)

        entry = pool.get_entry(0)
        assert entry.onboarding_complete is True

    def test_get_entry_invalid_index(self):
        """get_entry with invalid index should return None."""
        pool = CredentialPool()
        pool.add(self._create_entry(token="token1"))

        assert pool.get_entry(-1) is None
        assert pool.get_entry(99) is None
