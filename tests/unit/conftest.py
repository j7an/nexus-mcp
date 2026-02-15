# tests/unit/conftest.py
"""Shared fixtures for unit tests outside the runners/ directory."""

from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo


@pytest.fixture
def mock_cli_detection():
    """Mock CLI detection for tests that use GeminiRunner outside runners/.

    NOT autouse — only tests that explicitly use this fixture will get the mocks.
    This is for tests like test_extension_points.py that instantiate GeminiRunner
    via RunnerFactory.create("gemini").
    """
    with (
        patch("nexus_mcp.runners.gemini.detect_cli") as mock_detect,
        patch("nexus_mcp.runners.gemini.get_cli_version", return_value="0.12.0"),
    ):
        mock_detect.return_value = CLIInfo(found=True, path="/usr/bin/gemini")
        yield mock_detect
