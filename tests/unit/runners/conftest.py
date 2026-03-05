# tests/unit/runners/conftest.py
"""Shared fixtures for runner tests.

Autouse wrappers ensure mock_cli_detection and fast_retry_sleep are active
for every test in this directory without requiring explicit fixture requests.
The underlying fixtures live in tests/conftest.py to avoid duplication.
"""

import pytest


@pytest.fixture(autouse=True)
def _auto_mock_cli_detection(mock_cli_detection):
    """Auto-activate CLI detection mocking for all runner unit tests.

    GeminiRunner/CodexRunner.__init__ calls detect_cli() and get_cli_version().
    Mocking both prevents tests from requiring actual CLI binaries installed.
    RunnerFactory cache is cleared on teardown (via cli_detection_mocks).
    """
    yield mock_cli_detection


@pytest.fixture(autouse=True)
def _auto_fast_retry_sleep(fast_retry_sleep):
    """Auto-activate instant asyncio.sleep for all runner unit tests.

    Prevents real waiting during retry backoff loops in runner tests.
    Tests in tests/unit/test_process.py (timeout tests) are unaffected
    since they are outside this directory.
    """
