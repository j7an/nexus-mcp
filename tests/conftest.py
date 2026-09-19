# tests/conftest.py
"""Root conftest: shared fixtures visible to all test suites (unit + integration).

Fixtures here are available in tests/unit/ and tests/integration/ without
any additional imports.
"""

import os

os.environ.setdefault("FASTMCP_MCP_CAMELCASE_COMPAT", "false")

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Context
from mcp.shared.exceptions import MCPError

from nexus_mcp.runners.factory import RunnerFactory
from tests.fakes import FakeRunner
from tests.fixtures import cli_detection_mocks, make_fast_runtime_tuning


@pytest.fixture(autouse=True)
def _isolate_nexus_job_database(monkeypatch, tmp_path):
    """Give every test process a private durable-job database path."""
    monkeypatch.setenv("NEXUS_DB_PATH", str(tmp_path / "nexus.sqlite3"))


@pytest.fixture
def fast_job_runtime():
    """Use real-yielding short polls and zero retry delay for durable-job tests."""
    from nexus_mcp.mcp.runtime import runtime_provider

    with runtime_provider.override_tuning(make_fast_runtime_tuning()):
        yield


@pytest.fixture(autouse=True)
def _clean_runner_cache():
    """Clear RunnerFactory cache before and after each test."""
    RunnerFactory.clear_cache()
    yield
    RunnerFactory.clear_cache()


@pytest.fixture(autouse=True)
def _clean_preference_store():
    """Clear the in-process preference/tier store before and after each test."""
    from nexus_mcp.store import reset_store

    reset_store()
    yield
    reset_store()


@pytest.fixture
def fake_runner_registry():
    """Temporarily register the test-only fake runner."""
    original_registry = RunnerFactory._REGISTRY.copy()
    RunnerFactory.clear_cache()
    RunnerFactory._REGISTRY[FakeRunner.AGENT_NAME] = FakeRunner
    try:
        yield FakeRunner.AGENT_NAME
    finally:
        RunnerFactory._REGISTRY = original_registry
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
    mock.is_background_task = False
    mock.request_context.protocol_version = "2025-11-25"
    # By default, simulate a client that does not support elicitation.
    # Tests that need elicitation should configure mock.elicit explicitly.
    mock.elicit.side_effect = MCPError(code=-32600, message="not supported")
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
