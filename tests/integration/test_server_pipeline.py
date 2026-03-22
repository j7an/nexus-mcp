# tests/integration/test_server_pipeline.py
"""Integration tests for the full MCP server pipeline.

Tests exercise prompt() from server.py using the real Gemini CLI, and
verify server instructions are populated. Only Context is mocked for
progress assertions — all other components are real.

Imports the raw functions (not MCP-wrapped FunctionTool), matching the
pattern established in tests/unit/test_server.py.
"""

from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.server import prompt
from tests.fixtures import PING_PROMPT


class TestServerInstructionsSmokeTest:
    """Smoke tests for server instructions (no CLI required)."""

    def test_instructions_mention_gemini(self) -> None:
        """Server instructions should always mention 'gemini' runner."""
        from nexus_mcp.server import mcp

        assert mcp.instructions is not None
        assert "gemini" in mcp.instructions


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
        """prompt() reports runner-level progress (attempt + step-level calls).

        Single-task prompt uses unwrapped emitter: runner's progress/total pass
        through directly. Expect 5 calls: 1 attempt + 4 steps (happy path skips
        error-check step 3).
        """
        await prompt(
            cli="gemini",
            prompt=PING_PROMPT,
            ctx=ctx,
        )

        assert ctx.report_progress.await_count == 5
        # First call is attempt-level
        first_msg = ctx.report_progress.call_args_list[0].kwargs["message"]
        assert first_msg.startswith("Attempt ")
        # No task wrapper prefix (unwrapped emitter for single-task)
        messages = [c.kwargs["message"] for c in ctx.report_progress.call_args_list]
        assert all("Task '" not in m for m in messages)
