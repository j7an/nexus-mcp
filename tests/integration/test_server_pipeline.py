# tests/integration/test_server_pipeline.py
"""Integration tests for the full MCP server pipeline.

Tests exercise prompt() and list_agents() from server.py using the
real Gemini CLI. Only Context is mocked for progress assertions — all
other components are real.

Imports the raw functions (not MCP-wrapped FunctionTool), matching the
pattern established in tests/unit/test_server.py.
"""

from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError

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

    async def test_prompt_rejects_unsupported_agent(self) -> None:
        """prompt() should raise ToolError for unknown agent names."""
        with pytest.raises(ToolError, match="nonexistent_agent_12345"):
            await prompt(
                cli="nonexistent_agent_12345",
                prompt="test",
            )


@pytest.mark.integration
class TestServerPromptPipeline:
    """Full prompt() pipeline tests with real CLI execution."""

    @pytest.mark.slow
    async def test_prompt_returns_string(self, gemini_cli_available: str) -> None:  # noqa: ARG002
        """prompt() should return a non-empty string from real CLI output."""
        result = await prompt(
            cli="gemini",
            prompt=PING_PROMPT,
        )

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.slow
    async def test_prompt_reports_progress(self, gemini_cli_available: str, ctx: AsyncMock) -> None:  # noqa: ARG002
        """prompt() should call ctx.report_progress once with progress=1, total=1."""
        await prompt(
            cli="gemini",
            prompt=PING_PROMPT,
            ctx=ctx,
        )

        assert ctx.report_progress.await_count == 1
        call_kwargs = ctx.report_progress.call_args.kwargs
        assert call_kwargs["progress"] == 1
        assert call_kwargs["total"] == 1
