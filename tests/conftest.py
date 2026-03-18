# tests/conftest.py
"""Root conftest: shared fixtures visible to all test suites (unit + integration).

Fixtures here are available in tests/unit/ and tests/integration/ without
any additional imports.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Context

from nexus_mcp.runners.factory import RunnerFactory
from tests.fixtures import cli_detection_mocks


@pytest.fixture(autouse=True)
def _clean_runner_cache():
    """Clear RunnerFactory cache before and after each test.

    Config is now stateless (reads env vars fresh each call), so there is no
    singleton to reset. We still need to clear the runner cache to prevent
    runner instances from leaking between tests.
    """
    RunnerFactory.clear_cache()
    yield
    RunnerFactory.clear_cache()


@pytest.fixture
def ctx() -> AsyncMock:
    """Minimal mock for FastMCP Context DI sentinel.

    Context is None-defaulted in server functions, so tests that don't need
    it can omit it. This fixture provides a spec'd mock for tests that
    verify ctx.info() logging behavior.

    ctx.get_state returns None by default (no session state set).
    Tests that need specific session state can override: ctx.get_state.return_value = {...}
    """
    mock = AsyncMock(spec=Context)
    mock.get_state.return_value = None  # simulate empty session state
    return mock


@pytest.fixture
def mock_cli_detection():
    """Mock CLI detection so tests don't require real CLI binaries installed.

    NOT autouse — subdirectory conftest files wrap this as autouse for their
    respective test directories. Tests that need it explicitly can also request
    it by name.
    """
    with cli_detection_mocks() as mock:
        yield mock


@pytest.fixture
def fast_retry_sleep(monkeypatch):
    """Patch asyncio.sleep to be instant for retry backoff tests.

    NOT autouse — the runners/ conftest wraps this as autouse for unit tests.
    E2E tests have their own variant that patches _compute_backoff instead
    (to avoid busy-spinning the Docket worker's 250ms polling loop).
    """

    async def instant_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", instant_sleep)


@pytest.fixture
def mock_subprocess():
    """Patch asyncio.create_subprocess_exec at the process module boundary.

    All layers above the subprocess call run for real:
        tool/runner → build_command → run_subprocess → [MOCK]

    Clears RunnerFactory cache on teardown to prevent runner instances from
    leaking between tests.
    """
    with patch("nexus_mcp.process.asyncio.create_subprocess_exec") as mock_exec:
        yield mock_exec
    RunnerFactory.clear_cache()
