# tests/integration/test_server_pipeline.py
"""Integration tests for the full MCP server pipeline.

Tests exercise prompt_agent() and list_agents() from server.py using the
real Gemini CLI. Only Progress is mocked — all other components are real.

Imports the raw functions (not MCP-wrapped FunctionTool), matching the
pattern established in tests/unit/test_server.py.
"""

from unittest.mock import AsyncMock

import pytest

from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.server import list_agents, prompt_agent

# Minimal prompt to reduce API latency in slow tests.
_PING_PROMPT = "Reply with exactly the word 'pong'"


class TestServerListAgentsSmokeTest:
    """Smoke tests for list_agents() that require no CLI."""

    def test_list_agents_includes_gemini(self) -> None:
        """list_agents() should always include 'gemini' (no CLI required)."""
        agents = list_agents()

        assert "gemini" in agents


class TestServerPromptAgentValidation:
    """Input-validation tests for prompt_agent() that require no CLI."""

    async def test_prompt_agent_rejects_unsupported_agent(self, progress: AsyncMock) -> None:
        """prompt_agent() should raise UnsupportedAgentError for unknown agent names."""
        with pytest.raises(UnsupportedAgentError):
            await prompt_agent(
                agent="nonexistent_agent_12345",
                prompt="test",
                progress=progress,
            )


@pytest.mark.integration
class TestServerPromptAgentPipeline:
    """Full prompt_agent() pipeline tests with real CLI execution."""

    @pytest.mark.slow
    async def test_prompt_agent_returns_string(
        self, gemini_cli_available: str, progress: AsyncMock
    ) -> None:  # noqa: ARG002
        """prompt_agent() should return a non-empty string from real CLI output."""
        result = await prompt_agent(
            agent="gemini",
            prompt=_PING_PROMPT,
            progress=progress,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.slow
    async def test_prompt_agent_reports_progress(
        self, gemini_cli_available: str, progress: AsyncMock
    ) -> None:  # noqa: ARG002
        """prompt_agent() should call set_total once and 4 increments summing to 100."""
        await prompt_agent(
            agent="gemini",
            prompt=_PING_PROMPT,
            progress=progress,
        )

        progress.set_total.assert_called_once_with(100)
        assert progress.increment.call_count == 4
        assert sum(c.args[0] for c in progress.increment.call_args_list) == 100
