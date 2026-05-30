# tests/unit/test_cli_detector.py
"""Tests for CLI detection, version parsing, and capability checking."""

import logging
import subprocess
from unittest.mock import Mock, patch

import pytest

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
        with patch("nexus_mcp.cli_detector.shutil.which", return_value="/usr/bin/codex"):
            info = detect_cli("codex")
        assert info.found is True
        assert info.path == "/usr/bin/codex"

    def test_detect_cli_not_found(self):
        with patch("nexus_mcp.cli_detector.shutil.which", return_value=None):
            info = detect_cli("nonexistent-cli-12345")
        assert info.found is False
        assert info.path is None

    def test_detect_cli_returns_cli_info(self):
        with patch("nexus_mcp.cli_detector.shutil.which", return_value="/usr/bin/codex"):
            info = detect_cli("codex")
        assert isinstance(info, CLIInfo)
        assert hasattr(info, "found")
        assert hasattr(info, "path")
        assert hasattr(info, "version")

    def test_detect_opencode_server_without_binary(self):
        info = detect_cli("opencode_server")

        assert info.found is True
        assert info.path == "http"
        assert info.version is None


class TestGetCLIVersion:
    """Test get_cli_version() runs '<cli> --version' and extracts version."""

    @pytest.fixture(autouse=True)
    def mock_timeout(self):
        with patch("nexus_mcp.cli_detector.get_cli_detection_timeout", return_value=10):
            yield

    def test_get_cli_version_codex(self):
        with patch("nexus_mcp.cli_detector.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="codex-cli 0.107.0", stderr="", returncode=0)
            version = get_cli_version("codex")
        assert version == "0.107.0"
        mock_run.assert_called_once_with(
            ["codex", "--version"], capture_output=True, text=True, timeout=10
        )

    def test_get_cli_version_returns_none_on_failure(self):
        with patch("nexus_mcp.cli_detector.subprocess.run", side_effect=FileNotFoundError):
            version = get_cli_version("codex")
        assert version is None

    def test_get_cli_version_returns_none_on_empty_output(self):
        with patch("nexus_mcp.cli_detector.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="", stderr="", returncode=1)
            version = get_cli_version("codex")
        assert version is None

    def test_get_cli_version_parses_nonzero_exit(self):
        with patch("nexus_mcp.cli_detector.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="codex-cli 0.107.0", stderr="", returncode=1)
            version = get_cli_version("codex")
        assert version == "0.107.0"

    def test_get_cli_version_parses_stderr_fallback(self):
        with patch("nexus_mcp.cli_detector.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="", stderr="codex-cli 0.107.0", returncode=0)
            version = get_cli_version("codex")
        assert version == "0.107.0"

    def test_get_cli_version_returns_none_on_timeout(self):
        with patch(
            "nexus_mcp.cli_detector.subprocess.run",
            side_effect=subprocess.TimeoutExpired("codex", 10),
        ):
            version = get_cli_version("codex")
        assert version is None

    def test_get_cli_version_logs_warning_on_oserror(self, caplog):
        """OSError during version detection must emit a warning, not fail silently."""
        with (
            caplog.at_level(logging.WARNING, logger="nexus_mcp.cli_detector"),
            patch(
                "nexus_mcp.cli_detector.subprocess.run",
                side_effect=FileNotFoundError("codex"),
            ),
        ):
            version = get_cli_version("codex")
        assert version is None
        assert any("codex" in r.message for r in caplog.records)

    def test_get_cli_version_logs_warning_on_timeout(self, caplog):
        """TimeoutExpired during version detection must emit a warning."""
        with (
            caplog.at_level(logging.WARNING, logger="nexus_mcp.cli_detector"),
            patch(
                "nexus_mcp.cli_detector.subprocess.run",
                side_effect=subprocess.TimeoutExpired("codex", 10),
            ),
        ):
            version = get_cli_version("codex")
        assert version is None
        assert any("codex" in r.message for r in caplog.records)

    def test_get_cli_version_opencode_server_returns_none_without_subprocess(self):
        with patch("nexus_mcp.cli_detector.subprocess.run") as mock_run:
            version = get_cli_version("opencode_server")

        assert version is None
        mock_run.assert_not_called()

    def test_get_cli_version_unknown_cli_returns_none(self):
        """Unknown CLI name: subprocess succeeds but parse_version returns None.

        parse_version() uses patterns.get(cli) which returns None for unrecognised names,
        so the version string is never extracted regardless of subprocess output.
        """
        with patch("nexus_mcp.cli_detector.subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="some output 1.2.3", stderr="", returncode=0)
            version = get_cli_version("unknown_cli")
        assert version is None


class TestParseVersion:
    """Test parse_version() extracts semver from CLI output strings."""

    def test_parse_version_codex(self):
        assert parse_version("codex-cli 0.107.0", cli="codex") == "0.107.0"

    def test_parse_version_codex_version_format(self):
        assert parse_version("codex version 1.2.3", cli="codex") == "1.2.3"

    def test_parse_version_claude(self):
        assert parse_version("Claude Code CLI v2.5.0", cli="claude") == "2.5.0"

    def test_parse_version_invalid_output(self):
        assert parse_version("unknown format", cli="codex") is None

    def test_parse_version_opencode(self):
        assert parse_version("opencode v0.1.0", cli="opencode") == "0.1.0"

    def test_parse_version_opencode_without_v(self):
        assert parse_version("opencode 1.2.3", cli="opencode") == "1.2.3"

    def test_parse_version_unknown_cli(self):
        assert parse_version("some output", cli="unknown") is None


class TestSupportsJsonOutput:
    """Test version-based JSON output support checking."""

    def test_codex_always_supports_json(self):
        assert supports_json_output("codex", "1.0.0") is True

    def test_claude_always_supports_json(self):
        assert supports_json_output("claude", "1.0.0") is True

    def test_opencode_always_supports_json(self):
        assert supports_json_output("opencode", "0.1.0") is True

    def test_unknown_cli_no_json(self):
        assert supports_json_output("unknown", "1.0.0") is False


class TestGetCLICapabilities:
    """Test get_cli_capabilities() aggregates feature support."""

    def test_codex_capabilities(self):
        caps = get_cli_capabilities("codex", "0.107.0")
        assert caps.found is True
        assert caps.supports_json is True

    def test_claude_capabilities(self):
        caps = get_cli_capabilities("claude", "2.5.0")
        assert caps.found is True
        assert caps.supports_json is True

    def test_opencode_capabilities(self):
        caps = get_cli_capabilities("opencode", "0.1.0")
        assert caps.found is True
        assert caps.supports_json is True

    def test_cli_not_found(self):
        caps = get_cli_capabilities("nonexistent", None)
        assert caps.found is False
        assert caps.supports_json is False
