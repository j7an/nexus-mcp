# tests/e2e/conftest.py
"""Shared fixtures for E2E MCP protocol tests.

All tests in this directory call the server via FastMCP's in-process Client
(FastMCPTransport — no network). Mocking is done only at the subprocess
boundary, letting all layers above run for real:

    Client (JSON-RPC) → FastMCP server → tool functions → RunnerFactory
        → GeminiRunner → build_command → [MOCK subprocess]
"""

from unittest.mock import patch

import pytest
from fastmcp import Client

from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.server import mcp
from tests.fixtures import cli_detection_mocks


@pytest.fixture(autouse=True)
def mock_cli_detection():
    """Auto-mock CLI detection for all E2E tests.

    GeminiRunner.__init__ calls detect_cli() and get_cli_version().
    Mock both so tests don't require the real Gemini CLI installed.
    Clears RunnerFactory cache on teardown to prevent runner instances
    built under mocked CLI detection from leaking into subsequent tests.
    """
    with cli_detection_mocks() as mock:
        yield mock
    RunnerFactory.clear_cache()


@pytest.fixture
def mock_subprocess():
    """Patch asyncio.create_subprocess_exec at the process module boundary.

    All layers above the subprocess call run for real through the MCP protocol:
        Client (JSON-RPC) → FastMCP server → tool functions → RunnerFactory
            → GeminiRunner → build_command → run_subprocess → [MOCK]

    Clears RunnerFactory cache on teardown to prevent runner instances from leaking.
    """
    with patch("nexus_mcp.process.asyncio.create_subprocess_exec") as mock_exec:
        yield mock_exec
    RunnerFactory.clear_cache()


@pytest.fixture
async def mcp_client():
    """In-process MCP client using FastMCPTransport (no network).

    Provides a connected Client instance backed by the real FastMCP server.
    All JSON-RPC serialization, FastMCP DI injection of Progress/Context,
    and tool dispatch happen for real.

    Note: FastMCP's _lifespan_result_set flag is reset on teardown to prevent
    state pollution across tests. This flag can remain True if the lifespan
    exits via CancelledError (a FastMCP limitation), causing subsequent
    Client(mcp) connections to skip Docket initialization.
    """
    async with Client(mcp) as client:
        yield client
    # Reset FastMCP lifespan flag so the next test gets a fresh Docket setup.
    mcp._lifespan_result_set = False


@pytest.fixture
def fast_retry_sleep(monkeypatch):
    """Patch asyncio.sleep to be instant for retry-related E2E tests.

    Prevents real waiting during backoff delays when testing retry behavior.
    Only applied to tests that explicitly use this fixture (not autouse).
    """

    async def instant_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("asyncio.sleep", instant_sleep)
