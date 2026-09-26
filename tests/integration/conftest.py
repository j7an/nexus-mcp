"""Real-CLI fixtures; every test here is marked `integration` and is skipped in CI."""

import pytest

from nexus_mcp import backends


@pytest.fixture
def claude_installed() -> None:
    if not backends.installed("claude"):
        pytest.skip("claude extra not installed")


@pytest.fixture
def codex_installed() -> None:
    if not backends.installed("codex"):
        pytest.skip("codex extra not installed")
