# tests/integration/test_codex_runner.py
"""Integration tests for CodexRunner with real CLI binary.

Minimal set — only tests that require a real CLI binary.
Command building, NDJSON parsing, and retry logic are covered by unit tests.
"""

import pytest

from nexus_mcp.runners.codex import CodexRunner
from nexus_mcp.types import AgentResponse
from tests.fixtures import PING_PROMPT, make_prompt_request


@pytest.mark.integration
@pytest.mark.slow
class TestCodexRunnerIntegration:
    """Real CLI binary + real API call tests."""

    async def test_run_returns_valid_response(self, codex_runner: CodexRunner) -> None:
        """run() with real API returns AgentResponse with output and metadata."""
        request = make_prompt_request(cli="codex", prompt=PING_PROMPT)
        response = await codex_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert response.cli == "codex"
        assert len(response.output) > 0
        assert isinstance(response.metadata, dict)
