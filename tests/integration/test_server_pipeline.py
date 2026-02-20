# tests/integration/test_server_pipeline.py
"""Integration tests for the full MCP server pipeline.

Tests exercise prompt() and list_agents() from server.py using the
real Gemini CLI. Only Progress is mocked — all other components are real.

Imports the raw functions (not MCP-wrapped FunctionTool), matching the
pattern established in tests/unit/test_server.py.
"""

from unittest.mock import AsyncMock

import pytest

from nexus_mcp.server import list_agents, prompt
from tests.fixtures import PING_PROMPT


class TestServerListAgentsSmokeTest:
    """Smoke tests for list_agents() that require no CLI."""

    def test_list_agents_includes_gemini(self) -> None:
        """list_agents() should always include 'gemini' (no CLI required)."""
        agents = list_agents()

        assert "gemini" in agents


class TestServerPromptValidation:
    """Input-validation tests for prompt() that require no CLI."""

    async def test_prompt_rejects_unsupported_agent(self, progress: AsyncMock) -> None:
        """prompt() should raise RuntimeError for unknown agent names."""
        with pytest.raises(RuntimeError, match="nonexistent_agent_12345"):
            await prompt(
                agent="nonexistent_agent_12345",
                prompt="test",
                progress=progress,
            )


@pytest.mark.integration
class TestServerPromptPipeline:
    """Full prompt() pipeline tests with real CLI execution."""

    @pytest.mark.slow
    async def test_prompt_returns_string(
        self, gemini_cli_available: str, progress: AsyncMock
    ) -> None:  # noqa: ARG002
        """prompt() should return a non-empty string from real CLI output."""
        result = await prompt(
            agent="gemini",
            prompt=PING_PROMPT,
            progress=progress,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.slow
    async def test_prompt_reports_progress(
        self, gemini_cli_available: str, progress: AsyncMock
    ) -> None:  # noqa: ARG002
        """prompt() should call set_total(1) and increment once via batch_prompt."""
        await prompt(
            agent="gemini",
            prompt=PING_PROMPT,
            progress=progress,
        )

        progress.set_total.assert_called_once_with(1)
        assert progress.increment.call_count == 1
