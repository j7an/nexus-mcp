# tests/integration/test_opencode_runner.py
"""Integration tests for OpenCodeRunner with real CLI binary.

Minimal set — only tests that require a real CLI binary.
Command building, NDJSON parsing, and retry logic are covered by unit tests.
"""

import pytest

from nexus_mcp.runners.opencode import OpenCodeRunner
from nexus_mcp.types import AgentResponse
from tests.fixtures import PING_PROMPT, make_prompt_request


@pytest.mark.integration
@pytest.mark.slow
class TestOpenCodeRunnerIntegration:
    """Real CLI binary + real API call tests."""

    async def test_run_returns_valid_response(self, opencode_runner: OpenCodeRunner) -> None:
        """run() with real API returns AgentResponse with output and metadata."""
        request = make_prompt_request(cli="opencode", prompt=PING_PROMPT)
        response = await opencode_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert response.cli == "opencode"
        assert len(response.output) > 0
