# tests/unit/runners/conftest.py
"""Shared fixtures for runner tests.

The autouse wrapper ensures CLI detection is mocked for every test in this
directory without requiring explicit fixture requests. The underlying fixture
lives in tests/conftest.py to avoid duplication.
"""

import pytest


@pytest.fixture(autouse=True)
def _auto_mock_cli_detection(mock_cli_detection):
    """Auto-activate CLI detection mocking for all runner unit tests.

    Runner __init__ calls detect_cli() and get_cli_version().
    Mocking both prevents tests from requiring actual CLI binaries installed.
    RunnerFactory cache is cleared on teardown (via cli_detection_mocks).
    """
    yield mock_cli_detection
