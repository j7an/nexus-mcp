# tests/integration/conftest.py
"""Shared fixtures for integration tests.

No CLI mocking — all fixtures use real CLI binaries.
The root tests/conftest.py provides the shared `progress` fixture.
"""

import shutil

import pytest

from nexus_mcp.runners.claude import ClaudeRunner
from nexus_mcp.runners.codex import CodexRunner
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


@pytest.fixture(scope="session")
def codex_cli_available() -> str:
    """Skip all dependent tests if Codex CLI is not installed.

    Returns:
        Full path to the codex binary.
    """
    path = shutil.which("codex")
    if path is None:
        pytest.skip("Codex CLI not found in PATH — install to run integration tests")
    return path  # type: ignore[return-value]


@pytest.fixture
def codex_runner(codex_cli_available: str) -> CodexRunner:  # noqa: ARG001
    """Create a real CodexRunner per test.

    Fresh instance per test to avoid state leakage between tests.
    Depends on codex_cli_available to skip if CLI is absent.

    Uses gpt-5.2 as the default model because the Codex CLI's built-in
    default (gpt-5.3-codex) is not supported on standard ChatGPT accounts,
    and o4-mini is no longer supported on ChatGPT-authenticated accounts.
    """
    runner = CodexRunner()
    runner.default_model = "gpt-5.2"
    return runner


@pytest.fixture(scope="session")
def claude_cli_available() -> str:
    """Skip all dependent tests if Claude CLI is not installed.

    Returns:
        Full path to the claude binary.
    """
    path = shutil.which("claude")
    if path is None:
        pytest.skip("Claude CLI not found in PATH — install to run integration tests")
    return path  # type: ignore[return-value]


@pytest.fixture
def claude_runner(claude_cli_available: str) -> ClaudeRunner:  # noqa: ARG001
    """Create a real ClaudeRunner per test.

    Fresh instance per test to avoid state leakage between tests.
    Depends on claude_cli_available to skip if CLI is absent.
    """
    return ClaudeRunner()
