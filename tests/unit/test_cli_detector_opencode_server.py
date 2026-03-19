# tests/unit/test_cli_detector_opencode_server.py
"""Tests for opencode_server special-case in CLI detection."""

from nexus_mcp.cli_detector import (
    CLICapabilities,
    CLIInfo,
    detect_cli,
    get_cli_capabilities,
    get_cli_version,
)


class TestOpenCodeServerDetection:
    """opencode_server has no binary — detection is special-cased."""

    def test_detect_cli_returns_found_true(self):
        """detect_cli returns found=True with no path for opencode_server."""
        result = detect_cli("opencode_server")
        assert result == CLIInfo(found=True, path=None)

    def test_get_cli_version_returns_none(self):
        """get_cli_version returns None without calling subprocess."""
        result = get_cli_version("opencode_server")
        assert result is None

    def test_get_cli_capabilities_with_none_version(self):
        """Capabilities are found=False, supports_json=False (intentional — no binary)."""
        result = get_cli_capabilities("opencode_server", None)
        assert result == CLICapabilities(found=False, supports_json=False)
