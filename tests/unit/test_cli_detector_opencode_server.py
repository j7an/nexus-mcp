"""Tests for opencode_server CLI detection special-case."""

from nexus_mcp.cli_detector import CLIInfo, detect_cli, get_cli_version


class TestOpenCodeServerDetection:
    def test_detect_cli_returns_found_without_binary(self):
        """opencode_server has no binary — detection always returns found=True."""
        result = detect_cli("opencode_server")
        assert result == CLIInfo(found=True, path="http", version=None)

    def test_get_cli_version_returns_none(self):
        """opencode_server version is determined at runtime via health check, not CLI."""
        result = get_cli_version("opencode_server")
        assert result is None
