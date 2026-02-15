# tests/unit/runners/conftest.py
"""Shared fixtures for runner tests."""

from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo


@pytest.fixture(autouse=True)
def mock_cli_detection():
    """Auto-mock CLI detection for all runner tests.

    GeminiRunner.__init__ calls detect_cli() and get_cli_version().
    Mock both so tests don't require actual Gemini CLI installed.
    get_cli_capabilities is NOT mocked — it runs with the mocked version
    ("0.12.0" → supports_json=True), keeping existing command assertions valid.
    """
    with (
        patch("nexus_mcp.runners.gemini.detect_cli") as mock_detect,
        patch("nexus_mcp.runners.gemini.get_cli_version", return_value="0.12.0"),
    ):
        mock_detect.return_value = CLIInfo(found=True, path="/usr/bin/gemini")
        yield mock_detect
