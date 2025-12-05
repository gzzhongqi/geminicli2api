"""Tests for CLI credential management functionality."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cli import (
    CREDENTIALS_DIR,
    _get_user_email,
)


class TestCredentialStorage:
    """Tests for credential storage functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_credentials_dir = CREDENTIALS_DIR

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.cli.CREDENTIALS_DIR")
    def test_ensure_credentials_dir_creates_directory(self, mock_dir):
        """Should create credentials directory if it doesn't exist."""
        test_path = Path(self.temp_dir) / "test_creds"
        mock_dir.__truediv__ = lambda self, x: test_path / x
        mock_dir.mkdir = MagicMock()
        mock_dir.exists = MagicMock(return_value=False)
        mock_dir.parent = test_path.parent

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _ensure_credentials_dir

            result = _ensure_credentials_dir()
            assert result == test_path

    @patch("src.cli.CREDENTIALS_DIR")
    def test_get_credential_path(self, mock_dir):
        """Should return correct path for credential name."""
        test_path = Path(self.temp_dir)
        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _get_credential_path

            result = _get_credential_path("test_cred")
            assert result == test_path / "test_cred.json"

    @patch("src.cli.CREDENTIALS_DIR")
    def test_list_credentials_empty(self, mock_dir):
        """Should return empty list when no credentials exist."""
        test_path = Path(self.temp_dir) / "nonexistent"
        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _list_credentials

            result = _list_credentials()
            assert result == []

    @patch("src.cli.CREDENTIALS_DIR")
    def test_list_credentials_with_files(self, mock_dir):
        """Should return list of credentials when files exist."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create test credential files
        cred1 = {
            "email": "test1@example.com",
            "project_id": "project-1",
            "refresh_token": "token1",
            "created_at": "2024-01-01T00:00:00+00:00",
        }
        cred2 = {
            "email": "test2@example.com",
            "project_id": "project-2",
            "refresh_token": "token2",
            "created_at": "2024-01-02T00:00:00+00:00",
        }

        with open(test_path / "cred1.json", "w") as f:
            json.dump(cred1, f)
        with open(test_path / "cred2.json", "w") as f:
            json.dump(cred2, f)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _list_credentials

            result = _list_credentials()

            assert len(result) == 2
            names = [c["name"] for c in result]
            assert "cred1" in names
            assert "cred2" in names

    @patch("src.cli.CREDENTIALS_DIR")
    def test_generate_credential_name_first(self, mock_dir):
        """Should generate credential_1 when no credentials exist."""
        test_path = Path(self.temp_dir) / "empty"
        test_path.mkdir(parents=True, exist_ok=True)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _generate_credential_name

            result = _generate_credential_name()
            assert result == "credential_1"

    @patch("src.cli.CREDENTIALS_DIR")
    def test_generate_credential_name_increments(self, mock_dir):
        """Should increment number when credentials exist."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create existing credentials
        for name in ["credential_1.json", "credential_2.json"]:
            with open(test_path / name, "w") as f:
                json.dump({"refresh_token": "test"}, f)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _generate_credential_name

            result = _generate_credential_name()
            assert result == "credential_3"

    @patch("src.cli.CREDENTIALS_DIR")
    def test_generate_credential_name_fills_gaps(self, mock_dir):
        """Should fill gaps in credential numbering."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create credentials with gap (1, 3 - missing 2)
        for name in ["credential_1.json", "credential_3.json"]:
            with open(test_path / name, "w") as f:
                json.dump({"refresh_token": "test"}, f)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _generate_credential_name

            result = _generate_credential_name()
            assert result == "credential_2"


class TestSaveCredential:
    """Tests for saving credentials."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_mock_credentials(self) -> MagicMock:
        """Create mock Credentials object."""
        mock = MagicMock()
        mock.token = "test_token"
        mock.refresh_token = "test_refresh_token"
        mock.scopes = ["scope1", "scope2"]
        mock.expiry = datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)
        return mock

    @patch("src.cli.CREDENTIALS_DIR")
    def test_save_credential_creates_file(self, mock_dir):
        """Should create credential file with correct data."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        mock_creds = self._create_mock_credentials()

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _save_credential

            result = _save_credential(mock_creds, "test_cred", "test@example.com")

            assert result.exists()
            with open(result) as f:
                data = json.load(f)

            assert data["refresh_token"] == "test_refresh_token"
            assert data["email"] == "test@example.com"
            assert "created_at" in data

    @patch("src.cli.CREDENTIALS_DIR")
    def test_save_credential_without_email(self, mock_dir):
        """Should save credential without email field when not provided."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        mock_creds = self._create_mock_credentials()

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _save_credential

            result = _save_credential(mock_creds, "test_cred")

            with open(result) as f:
                data = json.load(f)

            assert "email" not in data


class TestRemoveCredential:
    """Tests for removing credentials."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.cli.CREDENTIALS_DIR")
    def test_remove_credential_success(self, mock_dir):
        """Should remove existing credential file."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create test credential
        cred_file = test_path / "test_cred.json"
        with open(cred_file, "w") as f:
            json.dump({"refresh_token": "test"}, f)

        assert cred_file.exists()

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _remove_credential

            result = _remove_credential("test_cred")

            assert result is True
            assert not cred_file.exists()

    @patch("src.cli.CREDENTIALS_DIR")
    def test_remove_credential_not_found(self, mock_dir):
        """Should return False when credential doesn't exist."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import _remove_credential

            result = _remove_credential("nonexistent")
            assert result is False


class TestGetUserEmail:
    """Tests for getting user email from credentials."""

    def test_get_user_email_success(self):
        """Should return email from userinfo API."""
        mock_creds = MagicMock()
        mock_creds.token = "test_token"

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"email": "test@example.com"}
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = _get_user_email(mock_creds)
            assert result == "test@example.com"

    def test_get_user_email_failure(self):
        """Should return None on API failure."""
        mock_creds = MagicMock()
        mock_creds.token = "test_token"

        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("API error")

            result = _get_user_email(mock_creds)
            assert result is None


class TestExportCommand:
    """Tests for export command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.cli.CREDENTIALS_DIR")
    def test_export_empty_credentials(self, mock_dir, capsys):
        """Should report no credentials when directory is empty."""
        test_path = Path(self.temp_dir) / "empty"

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_export

            args = MagicMock()
            args.docker = False
            args.output = None

            result = cmd_auth_export(args)
            assert result == 1

            captured = capsys.readouterr()
            assert "No credentials found" in captured.out

    @patch("src.cli.CREDENTIALS_DIR")
    def test_export_env_format(self, mock_dir, capsys):
        """Should export in .env format."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create test credential
        cred_data = {
            "refresh_token": "test_refresh_token",
            "email": "test@example.com",
        }
        with open(test_path / "cred1.json", "w") as f:
            json.dump(cred_data, f)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_export

            args = MagicMock()
            args.docker = False
            args.output = None

            result = cmd_auth_export(args)
            assert result == 0

            captured = capsys.readouterr()
            assert "GEMINI_CREDENTIALS_1" in captured.out
            assert "test_refresh_token" in captured.out

    @patch("src.cli.CREDENTIALS_DIR")
    def test_export_docker_format(self, mock_dir, capsys):
        """Should export in docker-compose format."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create test credential
        cred_data = {
            "refresh_token": "test_refresh_token",
            "email": "test@example.com",
        }
        with open(test_path / "cred1.json", "w") as f:
            json.dump(cred_data, f)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_export

            args = MagicMock()
            args.docker = True
            args.output = None

            result = cmd_auth_export(args)
            assert result == 0

            captured = capsys.readouterr()
            assert "environment:" in captured.out
            assert "GEMINI_CREDENTIALS_1" in captured.out

    @patch("src.cli.CREDENTIALS_DIR")
    def test_export_to_file(self, mock_dir):
        """Should export to file when output specified."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create test credential
        cred_data = {
            "refresh_token": "test_refresh_token",
            "email": "test@example.com",
        }
        with open(test_path / "cred1.json", "w") as f:
            json.dump(cred_data, f)

        output_file = test_path / "output.env"

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_export

            args = MagicMock()
            args.docker = False
            args.output = str(output_file)

            result = cmd_auth_export(args)
            assert result == 0
            assert output_file.exists()

            content = output_file.read_text()
            assert "GEMINI_CREDENTIALS_1" in content


class TestListCommand:
    """Tests for list command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.cli.CREDENTIALS_DIR")
    def test_list_empty(self, mock_dir, capsys):
        """Should show message when no credentials exist."""
        test_path = Path(self.temp_dir) / "empty"

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_list

            args = MagicMock()
            result = cmd_auth_list(args)
            assert result == 0

            captured = capsys.readouterr()
            assert "No credentials found" in captured.out

    @patch("src.cli.CREDENTIALS_DIR")
    def test_list_with_credentials(self, mock_dir, capsys):
        """Should list all credentials."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create test credentials
        for i in range(2):
            cred_data = {
                "email": f"test{i}@example.com",
                "refresh_token": f"token{i}",
                "created_at": "2024-01-01T00:00:00+00:00",
            }
            with open(test_path / f"cred{i}.json", "w") as f:
                json.dump(cred_data, f)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_list

            args = MagicMock()
            result = cmd_auth_list(args)
            assert result == 0

            captured = capsys.readouterr()
            assert "Found 2 credential(s)" in captured.out
            assert "test0@example.com" in captured.out
            assert "test1@example.com" in captured.out


class TestRemoveCommand:
    """Tests for remove command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("src.cli.CREDENTIALS_DIR")
    def test_remove_not_found(self, mock_dir, capsys):
        """Should report error when credential not found."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_remove

            args = MagicMock()
            args.name = "nonexistent"

            result = cmd_auth_remove(args)
            assert result == 1

            captured = capsys.readouterr()
            assert "not found" in captured.out

    @patch("src.cli.CREDENTIALS_DIR")
    @patch("builtins.input", return_value="y")
    def test_remove_with_confirmation(self, mock_input, mock_dir, capsys):
        """Should remove credential when user confirms."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create test credential
        cred_file = test_path / "test_cred.json"
        with open(cred_file, "w") as f:
            json.dump({"refresh_token": "test"}, f)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_remove

            args = MagicMock()
            args.name = "test_cred"

            result = cmd_auth_remove(args)
            assert result == 0
            assert not cred_file.exists()

    @patch("src.cli.CREDENTIALS_DIR")
    @patch("builtins.input", return_value="n")
    def test_remove_cancelled(self, mock_input, mock_dir, capsys):
        """Should not remove credential when user cancels."""
        test_path = Path(self.temp_dir)
        test_path.mkdir(exist_ok=True)

        # Create test credential
        cred_file = test_path / "test_cred.json"
        with open(cred_file, "w") as f:
            json.dump({"refresh_token": "test"}, f)

        with patch("src.cli.CREDENTIALS_DIR", test_path):
            from src.cli import cmd_auth_remove

            args = MagicMock()
            args.name = "test_cred"

            result = cmd_auth_remove(args)
            assert result == 0
            assert cred_file.exists()  # File should still exist

            captured = capsys.readouterr()
            assert "Cancelled" in captured.out
