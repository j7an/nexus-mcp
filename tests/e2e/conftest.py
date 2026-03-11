# tests/e2e/conftest.py
"""Shared fixtures for E2E MCP protocol tests.

All tests in this directory call the server via FastMCP's in-process Client
(FastMCPTransport — no network). Mocking is done only at the subprocess
boundary, letting all layers above run for real:

    Client (JSON-RPC) → FastMCP server → tool functions → RunnerFactory
        → GeminiRunner → build_command → [MOCK subprocess]
"""

import pytest
from fastmcp import Client

from nexus_mcp.server import mcp


@pytest.fixture(autouse=True)
def _auto_mock_cli_detection(mock_cli_detection):
    """Auto-activate CLI detection mocking for all E2E tests.

    Prevents tests from requiring real CLI binaries. RunnerFactory cache
    is cleared on teardown (via cli_detection_mocks in the root fixture).
    """
    yield mock_cli_detection


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
    try:
        async with Client(mcp) as client:
            yield client
    finally:
        # WORKAROUND: FastMCP _lifespan_result_set stays True after CancelledError,
        # causing subsequent Client(mcp) connections to skip Docket initialization.
        # Remove when upstream fixes lifespan state cleanup on CancelledError.
        mcp._lifespan_result_set = False


@pytest.fixture
def fast_retry_sleep(monkeypatch):
    """Eliminate retry backoff delays for retry-related E2E tests.

    Patches AbstractRunner._compute_backoff to return 0, so asyncio.sleep(0)
    is called instead of asyncio.sleep(up-to-2s). asyncio.sleep(0) properly
    yields to the event loop once (no busy-spin), while the real asyncio.sleep
    is left untouched so the Docket worker's 250ms polling interval functions
    normally.

    Patching the global asyncio.sleep instead would cause the Docket worker to
    busy-spin and starve the event loop, hanging the test indefinitely.
    """
    monkeypatch.setattr(
        "nexus_mcp.runners.base.AbstractRunner._compute_backoff",
        lambda self, attempt, retry_after: 0.0,
    )
