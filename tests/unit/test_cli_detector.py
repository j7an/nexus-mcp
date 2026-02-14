# tests/unit/test_cli_detector.py
"""Tests for CLI detection, version parsing, and capability checking."""

import subprocess
from unittest.mock import Mock, patch

from nexus_mcp.cli_detector import (
    CLIInfo,
    detect_cli,
    get_cli_capabilities,
    get_cli_version,
    parse_version,
    supports_json_output,
)


class TestDetectCLI:
    """Test detect_cli() finds CLI binaries via shutil.which."""

    def test_detect_cli_found(self):
        with patch("nexus_mcp.cli_detector.shutil.which", return_value="/usr/bin/gemini"):
            info = detect_cli("gemini")
        assert info.found is True
        assert info.path == "/usr/bin/gemini"

    def test_detect_cli_not_found(self):
        with patch("nexus_mcp.cli_detector.shutil.which", return_value=None):
            info = detect_cli("nonexistent-cli-12345")
        assert info.found is False
        assert info.path is None

    def test_detect_cli_returns_cli_info(self):
        with patch("nexus_mcp.cli_detector.shutil.which", return_value="/usr/bin/gemini"):
            info = detect_cli("gemini")
        assert isinstance(info, CLIInfo)
        assert hasattr(info, "found")
        assert hasattr(info, "path")
        assert hasattr(info, "version")


class TestGetCLIVersion:
    """Test get_cli_version() runs '<cli> --version' and extracts version."""

    def test_get_cli_version_gemini(self):
        with patch("nexus_mcp.cli_detector.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="Gemini CLI v0.12.0", returncode=0)
            version = get_cli_version("gemini")
        assert version == "0.12.0"
        mock_run.assert_called_once_with(
            ["gemini", "--version"], capture_output=True, text=True, timeout=10
        )

    def test_get_cli_version_returns_none_on_failure(self):
        with patch("nexus_mcp.cli_detector.subprocess.run", side_effect=FileNotFoundError):
            version = get_cli_version("gemini")
        assert version is None

    def test_get_cli_version_returns_none_on_nonzero_exit(self):
        with patch("nexus_mcp.cli_detector.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=1)
            version = get_cli_version("gemini")
        assert version is None

    def test_get_cli_version_returns_none_on_timeout(self):
        with patch(
            "nexus_mcp.cli_detector.subprocess.run",
            side_effect=subprocess.TimeoutExpired("gemini", 10),
        ):
            version = get_cli_version("gemini")
        assert version is None


class TestParseVersion:
    """Test parse_version() extracts semver from CLI output strings."""

    def test_parse_version_gemini(self):
        assert parse_version("Gemini CLI v0.6.0-preview.4", cli="gemini") == "0.6.0-preview.4"

    def test_parse_version_gemini_stable(self):
        assert parse_version("Gemini CLI v0.12.0", cli="gemini") == "0.12.0"

    def test_parse_version_codex(self):
        assert parse_version("codex version 1.2.3", cli="codex") == "1.2.3"

    def test_parse_version_claude(self):
        assert parse_version("Claude Code CLI v2.5.0", cli="claude") == "2.5.0"

    def test_parse_version_invalid_output(self):
        assert parse_version("unknown format", cli="gemini") is None

    def test_parse_version_unknown_cli(self):
        assert parse_version("some output", cli="unknown") is None


class TestSupportsJsonOutput:
    """Test version-based JSON output support checking."""

    def test_gemini_supports_json_modern(self):
        assert supports_json_output("gemini", "0.6.0") is True
        assert supports_json_output("gemini", "0.6.1") is True
        assert supports_json_output("gemini", "0.12.0") is True

    def test_gemini_preview_supports_json(self):
        """Pre-release of v0.6.0+ should still count as supporting JSON."""
        assert supports_json_output("gemini", "0.6.0-preview.4") is True
        assert supports_json_output("gemini", "0.12.0-rc.1") is True

    def test_gemini_old_no_json(self):
        assert supports_json_output("gemini", "0.5.0") is False
        assert supports_json_output("gemini", "0.5.9") is False
        assert supports_json_output("gemini", "0.5.9-preview.1") is False

    def test_codex_always_supports_json(self):
        assert supports_json_output("codex", "1.0.0") is True

    def test_claude_always_supports_json(self):
        assert supports_json_output("claude", "1.0.0") is True

    def test_unknown_cli_no_json(self):
        assert supports_json_output("unknown", "1.0.0") is False


class TestGetCLICapabilities:
    """Test get_cli_capabilities() aggregates feature support."""

    def test_gemini_modern(self):
        caps = get_cli_capabilities("gemini", "0.12.0")
        assert caps.found is True
        assert caps.supports_json is True
        assert caps.supports_model_flag is True
        assert caps.fallback_to_text is False

    def test_gemini_old(self):
        caps = get_cli_capabilities("gemini", "0.5.0")
        assert caps.found is True
        assert caps.supports_json is False
        assert caps.fallback_to_text is True

    def test_cli_not_found(self):
        caps = get_cli_capabilities("nonexistent", None)
        assert caps.found is False
        assert caps.supports_json is False
