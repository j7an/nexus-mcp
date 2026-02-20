# tests/unit/runners/conftest.py
"""Shared fixtures for runner tests."""

import pytest

from tests.fixtures import cli_detection_mocks


@pytest.fixture(autouse=True)
def mock_cli_detection():
    """Auto-mock CLI detection for all runner tests.

    GeminiRunner.__init__ calls detect_cli() and get_cli_version().
    Mock both so tests don't require actual Gemini CLI installed.
    get_cli_capabilities is NOT mocked — it runs with the mocked version
    ("0.12.0" → supports_json=True), keeping existing command assertions valid.
    """
    with cli_detection_mocks() as mock:
        yield mock
