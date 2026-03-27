# tests/integration/test_claude_runner.py
"""Integration tests for ClaudeRunner with real CLI binary.

Minimal set — only tests that require a real CLI binary.
Command building, JSON parsing, and retry logic are covered by unit tests.
"""

import pytest

from nexus_mcp.runners.claude import ClaudeRunner
from nexus_mcp.types import AgentResponse
from tests.fixtures import PING_PROMPT, make_prompt_request


@pytest.mark.integration
@pytest.mark.slow
class TestClaudeRunnerIntegration:
    """Real CLI binary + real API call tests."""

    async def test_run_returns_valid_response(self, claude_runner: ClaudeRunner) -> None:
        """run() with real API returns AgentResponse with output and metadata."""
        request = make_prompt_request(cli="claude", prompt=PING_PROMPT)
        response = await claude_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert response.cli == "claude"
        assert len(response.output) > 0
        assert isinstance(response.metadata, dict)
        assert "cost_usd" in response.metadata
        assert "duration_ms" in response.metadata

    async def test_invalid_model_raises_or_recovers(self, claude_runner: ClaudeRunner) -> None:
        """run() with nonexistent model raises SubprocessError or recovers gracefully."""
        from nexus_mcp.exceptions import SubprocessError

        request = make_prompt_request(cli="claude", prompt="ping", model="nonexistent-model-xyz-99")
        try:
            response = await claude_runner.run(request)
            assert response.metadata.get("recovered_from_error") is True, (
                f"Expected SubprocessError or recovered_from_error=True, got: {response.output!r}"
            )
        except SubprocessError:
            pass
