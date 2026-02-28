# tests/unit/runners/conftest.py
"""Shared fixtures for runner tests."""

import pytest

from nexus_mcp.runners.factory import RunnerFactory
from tests.fixtures import cli_detection_mocks


@pytest.fixture(autouse=True)
def mock_cli_detection():
    """Auto-mock CLI detection for all runner tests.

    GeminiRunner.__init__ calls detect_cli() and get_cli_version().
    Mock both so tests don't require actual Gemini CLI installed.
    get_cli_capabilities is NOT mocked — it runs with the mocked version
    ("0.12.0" → supports_json=True), keeping existing command assertions valid.

    Also clears the RunnerFactory cache on teardown so cached runner instances
    built under mocked CLI detection don't leak into subsequent tests.
    """
    with cli_detection_mocks() as mock:
        yield mock
    RunnerFactory.clear_cache()


@pytest.fixture(autouse=True)
def fast_retry_sleep(monkeypatch):
    """Patch asyncio.sleep to be instant for all runner unit tests.

    Prevents real waiting during the retry backoff loop. Tests in
    tests/unit/test_process.py (timeout tests) are unaffected since
    they are outside this directory.
    """

    async def instant_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", instant_sleep)
