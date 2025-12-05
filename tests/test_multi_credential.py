"""Tests for multi-credential loading functionality."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from src.services.auth import (
    CredentialEntry,
    _load_credential_entry_from_file,
    _load_credential_entry_from_json,
    _load_multiple_credentials_from_env,
    _load_multiple_credentials_from_files,
    initialize_credential_pool,
)


class TestLoadCredentialEntryFromJson:
    """Tests for _load_credential_entry_from_json function."""

    def test_valid_json_creates_entry(self):
        """Valid JSON should create a CredentialEntry."""
        json_str = json.dumps(
            {
                "refresh_token": "test_refresh_token",
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
            }
        )

        with patch("src.services.auth._create_credentials_from_data") as mock_create:
            mock_creds = MagicMock()
            mock_creds.expired = False
            mock_create.return_value = mock_creds

            entry = _load_credential_entry_from_json(json_str, "test_source")

            assert entry is not None
            assert entry.source == "test_source"
            assert entry.from_env is True

    def test_invalid_json_returns_none(self):
        """Invalid JSON should return None."""
        entry = _load_credential_entry_from_json("not valid json", "test_source")
        assert entry is None

    def test_missing_refresh_token_returns_none(self):
        """JSON without refresh_token should return None."""
        json_str = json.dumps(
            {
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
            }
        )

        entry = _load_credential_entry_from_json(json_str, "test_source")
        assert entry is None

    def test_extracts_project_id(self):
        """Should extract project_id from JSON if present."""
        json_str = json.dumps(
            {
                "refresh_token": "test_refresh_token",
                "client_id": "test_client_id",
                "client_secret": "test_client_secret",
                "project_id": "my-project-123",
            }
        )

        with patch("src.services.auth._create_credentials_from_data") as mock_create:
            mock_creds = MagicMock()
            mock_creds.expired = False
            mock_create.return_value = mock_creds

            entry = _load_credential_entry_from_json(json_str, "test_source")

            assert entry is not None
            assert entry.project_id == "my-project-123"


class TestLoadCredentialEntryFromFile:
    """Tests for _load_credential_entry_from_file function."""

    def test_valid_file_creates_entry(self):
        """Valid credentials file should create a CredentialEntry."""
        creds_data = {
            "refresh_token": "test_refresh_token",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(creds_data, f)
            temp_path = f.name

        try:
            with patch(
                "src.services.auth._create_credentials_from_data"
            ) as mock_create:
                mock_creds = MagicMock()
                mock_creds.expired = False
                mock_create.return_value = mock_creds

                entry = _load_credential_entry_from_file(temp_path, "test_source")

                assert entry is not None
                assert entry.source == "test_source"
                assert entry.from_env is False
        finally:
            os.unlink(temp_path)

    def test_nonexistent_file_returns_none(self):
        """Non-existent file should return None."""
        entry = _load_credential_entry_from_file("/nonexistent/path.json", "test")
        assert entry is None

    def test_invalid_json_file_returns_none(self):
        """File with invalid JSON should return None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            temp_path = f.name

        try:
            entry = _load_credential_entry_from_file(temp_path, "test_source")
            assert entry is None
        finally:
            os.unlink(temp_path)


class TestLoadMultipleCredentialsFromEnv:
    """Tests for _load_multiple_credentials_from_env function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_env_vars_returns_empty(self):
        """No GEMINI_CREDENTIALS_N vars should return empty list."""
        entries = _load_multiple_credentials_from_env()
        assert entries == []

    @patch.dict(
        os.environ,
        {
            "GEMINI_CREDENTIALS_1": '{"refresh_token":"token1"}',
            "GEMINI_CREDENTIALS_2": '{"refresh_token":"token2"}',
            "UNRELATED_VAR": "something",
        },
        clear=True,
    )
    def test_loads_indexed_env_vars(self):
        """Should load GEMINI_CREDENTIALS_N environment variables."""
        with patch("src.services.auth._load_credential_entry_from_json") as mock_load:
            mock_entry = MagicMock()
            mock_load.return_value = mock_entry

            entries = _load_multiple_credentials_from_env()

            assert len(entries) == 2
            assert mock_load.call_count == 2

    @patch.dict(
        os.environ,
        {
            "GEMINI_CREDENTIALS_3": '{"refresh_token":"token3"}',
            "GEMINI_CREDENTIALS_1": '{"refresh_token":"token1"}',
            "GEMINI_CREDENTIALS_2": '{"refresh_token":"token2"}',
        },
        clear=True,
    )
    def test_loads_in_index_order(self):
        """Should load credentials in numeric index order."""
        call_order = []

        def track_calls(json_str, source):
            call_order.append(source)
            return MagicMock()

        with patch(
            "src.services.auth._load_credential_entry_from_json",
            side_effect=track_calls,
        ):
            _load_multiple_credentials_from_env()

            # Should be sorted by index
            assert call_order == [
                "env:GEMINI_CREDENTIALS_1",
                "env:GEMINI_CREDENTIALS_2",
                "env:GEMINI_CREDENTIALS_3",
            ]


class TestLoadMultipleCredentialsFromFiles:
    """Tests for _load_multiple_credentials_from_files function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_env_var_returns_empty(self):
        """No GEMINI_CREDENTIAL_FILES var should return empty list."""
        entries = _load_multiple_credentials_from_files()
        assert entries == []

    @patch.dict(os.environ, {"GEMINI_CREDENTIAL_FILES": ""}, clear=True)
    def test_empty_env_var_returns_empty(self):
        """Empty GEMINI_CREDENTIAL_FILES should return empty list."""
        entries = _load_multiple_credentials_from_files()
        assert entries == []

    @patch.dict(
        os.environ,
        {"GEMINI_CREDENTIAL_FILES": "cred1.json,cred2.json,cred3.json"},
        clear=True,
    )
    def test_loads_comma_separated_files(self):
        """Should load comma-separated file paths."""
        with patch("src.services.auth._load_credential_entry_from_file") as mock_load:
            mock_entry = MagicMock()
            mock_load.return_value = mock_entry

            entries = _load_multiple_credentials_from_files()

            assert len(entries) == 3
            assert mock_load.call_count == 3

    @patch.dict(
        os.environ,
        {"GEMINI_CREDENTIAL_FILES": "cred1.json, cred2.json , cred3.json"},
        clear=True,
    )
    def test_handles_whitespace_in_paths(self):
        """Should handle whitespace around file paths."""
        with patch("src.services.auth._load_credential_entry_from_file") as mock_load:
            mock_entry = MagicMock()
            mock_load.return_value = mock_entry

            entries = _load_multiple_credentials_from_files()

            # All 3 should be loaded with trimmed paths
            assert len(entries) == 3


class TestInitializeCredentialPool:
    """Tests for initialize_credential_pool function."""

    def _create_mock_entry(self) -> CredentialEntry:
        """Create a mock CredentialEntry with proper attributes."""
        mock_creds = MagicMock()
        mock_creds.token = "test_token"
        mock_creds.expired = False
        return CredentialEntry(
            credentials=mock_creds,
            source="test",
            project_id=None,
        )

    @patch("src.services.auth._credential_pool", None)
    @patch("src.services.auth._load_multiple_credentials_from_env")
    @patch("src.services.auth._load_multiple_credentials_from_files")
    def test_loads_from_multiple_sources(self, mock_files, mock_env):
        """Should load credentials from all sources."""
        mock_entry1 = self._create_mock_entry()
        mock_entry2 = self._create_mock_entry()
        mock_env.return_value = [mock_entry1]
        mock_files.return_value = [mock_entry2]

        pool = initialize_credential_pool()

        assert pool.size() == 2
        mock_env.assert_called_once()
        mock_files.assert_called_once()

    @patch("src.services.auth._credential_pool", None)
    @patch("src.services.auth._load_multiple_credentials_from_env")
    @patch("src.services.auth._load_multiple_credentials_from_files")
    @patch.dict(os.environ, {"GEMINI_CREDENTIALS": '{"refresh_token":"single"}'})
    def test_fallback_to_single_credential(self, mock_files, mock_env):
        """Should fallback to single GEMINI_CREDENTIALS if no indexed vars."""
        mock_env.return_value = []
        mock_files.return_value = []

        with patch("src.services.auth._load_credential_entry_from_json") as mock_load:
            mock_entry = self._create_mock_entry()
            mock_load.return_value = mock_entry

            pool = initialize_credential_pool()

            assert pool.size() == 1
            mock_load.assert_called_once()

    @patch("src.services.auth._credential_pool", None)
    @patch("src.services.auth._load_multiple_credentials_from_env")
    @patch("src.services.auth._load_multiple_credentials_from_files")
    @patch.dict(os.environ, {}, clear=True)
    def test_fallback_to_credential_file(self, mock_files, mock_env):
        """Should fallback to default credential file if no env vars."""
        mock_env.return_value = []
        mock_files.return_value = []

        with patch("src.services.auth._load_credential_entry_from_file") as mock_load:
            mock_entry = self._create_mock_entry()
            mock_load.return_value = mock_entry

            pool = initialize_credential_pool()

            assert pool.size() == 1
            mock_load.assert_called_once()
