# tests/integration/conftest.py
"""Shared fixtures for integration tests.

No CLI mocking — all fixtures use real CLI binaries.
The root tests/conftest.py provides the shared `progress` fixture.
"""

import shutil

import pytest

from nexus_mcp.runners.gemini import GeminiRunner


@pytest.fixture(scope="session")
def gemini_cli_available() -> str:
    """Skip all dependent tests if Gemini CLI is not installed.

    Returns:
        Full path to the gemini binary.
    """
    path = shutil.which("gemini")
    if path is None:
        pytest.skip("Gemini CLI not found in PATH — install to run integration tests")
    return path  # type: ignore[return-value]  # pytest.skip() raises, so path is never None here


@pytest.fixture
def gemini_runner(gemini_cli_available: str) -> GeminiRunner:  # noqa: ARG001
    """Create a real GeminiRunner per test.

    Fresh instance per test to avoid state leakage between tests.
    Depends on gemini_cli_available to skip if CLI is absent.
    """
    return GeminiRunner()
