"""
Utility functions for Geminicli2api.
"""

import platform
from typing import Any, Dict, Optional

from .config import CLI_VERSION


def get_user_agent() -> str:
    """Generate User-Agent string matching gemini-cli format."""
    system = platform.system()
    arch = platform.machine()
    return f"GeminiCLI/{CLI_VERSION} ({system}; {arch})"


def get_platform_string() -> str:
    """Generate platform string matching gemini-cli format."""
    system = platform.system().upper()
    arch = platform.machine().upper()

    platform_map = {
        ("DARWIN", "ARM64"): "DARWIN_ARM64",
        ("DARWIN", "AARCH64"): "DARWIN_ARM64",
        ("DARWIN", "X86_64"): "DARWIN_AMD64",
        ("LINUX", "ARM64"): "LINUX_ARM64",
        ("LINUX", "AARCH64"): "LINUX_ARM64",
        ("LINUX", "X86_64"): "LINUX_AMD64",
        ("WINDOWS", "AMD64"): "WINDOWS_AMD64",
        ("WINDOWS", "X86_64"): "WINDOWS_AMD64",
    }

    return platform_map.get((system, arch), f"{system}_AMD64")


def get_client_metadata(project_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate client metadata for API requests.

    Args:
        project_id: Optional Google Cloud project ID

    Returns:
        Client metadata dictionary
    """
    return {
        "ideType": "IDE_UNSPECIFIED",
        "platform": get_platform_string(),
        "pluginType": "GEMINI",
        "duetProject": project_id,
    }
