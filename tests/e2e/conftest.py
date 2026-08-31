# tests/e2e/conftest.py
"""Shared fixtures for E2E MCP protocol tests.

All tests in this directory call the server via FastMCP's in-process Client
(FastMCPTransport — no network). Mocking is done only at the subprocess
boundary, letting all layers above run for real:

    Client (JSON-RPC) → FastMCP server → tool functions → RunnerFactory
        → runner → build_command → [MOCK subprocess]
"""

import contextlib

import pytest
from fastmcp import Client

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.server import mcp
from nexus_mcp.store import PREFERENCES_COLLECTION, PREFERENCES_KEY, TIERS_COLLECTION, TIERS_KEY
from tests.fakes import FakeRunner


@pytest.fixture(autouse=True)
async def _clean_preferences_store():
    """Clear persistent preferences before each E2E test.

    Preferences now persist across MCP sessions via the backing store.
    Without cleanup, preferences set in one test leak into subsequent tests.
    """
    store = mcp._state_store
    with contextlib.suppress(Exception):
        await store.delete(key=PREFERENCES_KEY, collection=PREFERENCES_COLLECTION)
    yield
    with contextlib.suppress(Exception):
        await store.delete(key=PREFERENCES_KEY, collection=PREFERENCES_COLLECTION)


@pytest.fixture(autouse=True)
async def _clean_tiers_store():
    """Clear persisted tier data between E2E tests."""
    store = mcp._state_store
    with contextlib.suppress(Exception):
        await store.delete(key=TIERS_KEY, collection=TIERS_COLLECTION)
    yield
    with contextlib.suppress(Exception):
        await store.delete(key=TIERS_KEY, collection=TIERS_COLLECTION)


@pytest.fixture(autouse=True)
def _auto_mock_cli_detection(mock_cli_detection, monkeypatch):
    """Auto-activate CLI detection mocking for all E2E tests.

    Prevents tests from requiring real CLI binaries. RunnerFactory cache
    is cleared on teardown (via cli_detection_mocks in the root fixture).
    """
    monkeypatch.setattr(
        "nexus_mcp.legacy.runner_backend.detect_cli",
        lambda _backend: CLIInfo(found=True, path="/test/cli"),
    )
    monkeypatch.setattr("nexus_mcp.legacy.runner_backend.get_cli_version", lambda _backend: "test")
    yield mock_cli_detection


@pytest.fixture
async def mcp_client(request):
    """In-process MCP client using FastMCPTransport (no network).

    Provides a connected Client instance backed by the real FastMCP server.
    All JSON-RPC serialization, FastMCP DI injection of Progress/Context,
    and tool dispatch happen for real.

    Note: FastMCP's _lifespan_result_set flag is reset on teardown to prevent
    state pollution across tests. This flag can remain True if the lifespan
    exits via CancelledError (a FastMCP limitation), causing subsequent
    Client(mcp) connections to skip Docket initialization.
    """
    needs_fake_backend = (
        "fake_runner_registry" in request.fixturenames
        or request.node.path.name == "test_middleware_integration.py"
    )
    original_registry = RunnerFactory._REGISTRY.copy()
    if needs_fake_backend:
        RunnerFactory.clear_cache()
        RunnerFactory._REGISTRY[FakeRunner.AGENT_NAME] = FakeRunner
    try:
        async with Client(mcp) as client:
            yield client
    finally:
        # WORKAROUND: FastMCP _lifespan_result_set stays True after CancelledError,
        # causing subsequent Client(mcp) connections to skip Docket initialization.
        # Remove when upstream fixes lifespan state cleanup on CancelledError.
        mcp._lifespan_result_set = False
        RunnerFactory._REGISTRY = original_registry
        RunnerFactory.clear_cache()


@pytest.fixture
async def job_mcp_client(fake_runner_registry, fast_job_runtime, monkeypatch):
    """Connected client whose lifespan sees the registered fake legacy runner."""
    from nexus_mcp.legacy import runner_backend

    original_detect = runner_backend.detect_cli
    original_version = runner_backend.get_cli_version
    monkeypatch.setattr(
        runner_backend,
        "detect_cli",
        lambda backend: (
            CLIInfo(found=True, path="/test/fake")
            if backend == fake_runner_registry
            else original_detect(backend)
        ),
    )
    monkeypatch.setattr(
        runner_backend,
        "get_cli_version",
        lambda backend: "test" if backend == fake_runner_registry else original_version(backend),
    )
    del fast_job_runtime
    try:
        async with Client(mcp) as client:
            yield client
    finally:
        mcp._lifespan_result_set = False


@pytest.fixture
def fast_job_mcp_client(fast_job_runtime, mcp_client):
    """Install fast durable-job tuning before the ordinary client enters lifespan."""
    del fast_job_runtime
    return mcp_client


@pytest.fixture
async def progress_mcp_client(fast_job_runtime):
    """Connected client that records compatibility progress notifications."""
    del fast_job_runtime
    progress_events: list[tuple[float, float | None, str | None]] = []

    async def record_progress(progress: float, total: float | None, message: str | None) -> None:
        progress_events.append((progress, total, message))

    try:
        async with Client(mcp, progress_handler=record_progress) as client:
            yield client, progress_events
    finally:
        mcp._lifespan_result_set = False
