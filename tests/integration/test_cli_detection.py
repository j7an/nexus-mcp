# tests/integration/test_cli_detection.py
"""Fallback tests for CLI detection functions (no CLI binary required).

The real CLI detection tests are consolidated into test_gemini_runner.py
(TestGeminiCLISmoke) to avoid redundancy.
"""

from nexus_mcp.cli_detector import detect_cli, get_cli_version


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
