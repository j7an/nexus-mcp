# tests/integration/test_cli_detection.py
"""Integration tests for CLI detection functions.

Validates detect_cli(), get_cli_version(), and get_cli_capabilities()
against the real installed Gemini CLI binary.
"""

import re

import pytest

from nexus_mcp.cli_detector import detect_cli, get_cli_capabilities, get_cli_version

# Semver pattern: MAJOR.MINOR.PATCH (optional pre-release suffix handled by parse_version)
_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+")


class TestDetectCLIFallback:
    """Pure-Python tests for detect_cli() that require no installed CLI."""

    def test_detect_nonexistent_cli_returns_not_found(self) -> None:
        """detect_cli() with unknown binary name should return found=False gracefully."""
        result = detect_cli("nonexistent_cli_that_does_not_exist_12345")

        assert result.found is False
        assert result.path is None


class TestGetCLIVersionFallback:
    """Pure-Python tests for get_cli_version() that require no installed CLI."""

    def test_get_version_nonexistent_cli_returns_none(self) -> None:
        """get_cli_version() for missing CLI should return None without raising."""
        version = get_cli_version("nonexistent_cli_that_does_not_exist_12345")

        assert version is None


@pytest.mark.integration
class TestDetectCLIReal:
    """Validate detect_cli() against real PATH entries."""

    def test_detect_gemini_finds_real_binary(self, gemini_cli_available: str) -> None:
        """detect_cli('gemini') should find the installed binary and return its path."""
        result = detect_cli("gemini")

        assert result.found is True
        assert result.path is not None
        assert result.path == gemini_cli_available


@pytest.mark.integration
class TestGetCLIVersionReal:
    """Validate get_cli_version() runs real subprocesses."""

    def test_get_gemini_version_returns_semver(self, gemini_cli_available: str) -> None:  # noqa: ARG002
        """get_cli_version('gemini') should parse a valid semver from the installed CLI."""
        version = get_cli_version("gemini")

        assert version is not None
        assert _SEMVER_PATTERN.match(version), f"Expected semver format, got: {version!r}"


@pytest.mark.integration
class TestGetCLICapabilitiesReal:
    """Validate get_cli_capabilities() with real version data."""

    def test_gemini_capabilities_with_real_version(self, gemini_cli_available: str) -> None:  # noqa: ARG002
        """get_cli_capabilities() should populate CLICapabilities from real version."""
        version = get_cli_version("gemini")
        assert version is not None, (
            "get_cli_version('gemini') returned None — version output format may have changed"
        )
        capabilities = get_cli_capabilities("gemini", version)

        assert capabilities.found is True
        assert isinstance(capabilities.supports_json, bool)
        assert isinstance(capabilities.supports_model_flag, bool)
        assert isinstance(capabilities.fallback_to_text, bool)

    def test_gemini_modern_version_supports_json(self, gemini_cli_available: str) -> None:  # noqa: ARG002
        """Installed Gemini CLI should be >= 0.6.0 and support JSON output."""
        version = get_cli_version("gemini")
        assert version is not None, (
            "get_cli_version('gemini') returned None — version output format may have changed"
        )
        capabilities = get_cli_capabilities("gemini", version)

        # Integration environment must have a modern CLI (>= 0.6.0)
        assert capabilities.supports_json is True, (
            f"Expected Gemini CLI >= 0.6.0 to support JSON output, got version: {version}"
        )
