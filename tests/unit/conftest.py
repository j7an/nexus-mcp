# tests/unit/conftest.py
"""Shared fixtures for unit tests outside the runners/ directory."""

import pytest

from tests.fixtures import cli_detection_mocks


@pytest.fixture
def mock_cli_detection():
    """Mock CLI detection for tests that use GeminiRunner outside runners/.

    NOT autouse — only tests that explicitly use this fixture will get the mocks.
    This is for tests like test_extension_points.py that instantiate GeminiRunner
    via RunnerFactory.create("gemini").
    """
    with cli_detection_mocks() as mock:
        yield mock
