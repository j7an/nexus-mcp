# tests/conftest.py
"""Root conftest: shared fixtures visible to all test suites (unit + integration).

Fixtures here are available in tests/unit/ and tests/integration/ without
any additional imports.
"""

from unittest.mock import AsyncMock

import pytest
from fastmcp.server.dependencies import InMemoryProgress


@pytest.fixture
def progress() -> AsyncMock:
    """Minimal mock for FastMCP Progress DI sentinel.

    Progress is the only mock in the integration suite — it cannot be
    provided by FastMCP outside an active MCP server context.

    Uses spec=InMemoryProgress (the concrete ProgressLike implementation that
    FastMCP v3 DI injects in non-Docket contexts) so that calls to non-existent
    methods raise AttributeError instead of silently succeeding.
    """
    return AsyncMock(spec=InMemoryProgress)
