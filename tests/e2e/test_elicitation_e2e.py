# tests/e2e/test_elicitation_e2e.py
"""E2E tests for ElicitationGuard graceful skip (elicit=False).

Tests the full MCP protocol stack with elicitation disabled:
- prompt tool with elicit=False succeeds when cli is provided
- prompt tool with elicit=False and no cli raises a ToolError
- batch_prompt tool with elicit=False succeeds when all tasks have cli set

Mock boundary: asyncio.create_subprocess_exec only.
All layers above run for real, including JSON-RPC dispatch.
"""

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.elicitation import ElicitationGuard
from tests.fixtures import GEMINI_JSON_RESPONSE, create_mock_process


@pytest.fixture(autouse=True)
def _reset_elicitation_cache():
    """Reset ElicitationGuard class-level cache before and after each test.

    The _elicitation_available cache persists across tests within the same
    process. Without this reset, a test that triggers McpError (elicitation
    unavailable) would pollute subsequent tests that expect elicitation to
    be uncached.
    """
    ElicitationGuard._elicitation_available = None
    yield
    ElicitationGuard._elicitation_available = None


@pytest.mark.e2e
class TestElicitationGracefulSkip:
    """Verify prompt and batch_prompt tools with elicit=False through full MCP protocol."""

    async def test_prompt_with_elicit_false(self, mock_subprocess, mcp_client):
        """prompt with cli set and elicit=False succeeds without any elicitation."""
        mock_subprocess.return_value = create_mock_process(stdout=GEMINI_JSON_RESPONSE)

        result = await mcp_client.call_tool(
            "prompt",
            {"cli": "gemini", "prompt": "Hello world test", "elicit": False},
        )

        assert result.is_error is False
        assert result.data == "test output"

    async def test_prompt_without_cli_and_elicit_false_errors(self, mcp_client):
        """prompt without cli and elicit=False raises ToolError about cli being required."""
        with pytest.raises(ToolError, match="cli is required"):
            await mcp_client.call_tool(
                "prompt",
                {"prompt": "Hello world test", "elicit": False},
            )

    async def test_batch_prompt_with_elicit_false(self, mock_subprocess, mcp_client):
        """batch_prompt with all tasks having cli set and elicit=False succeeds."""
        mock_subprocess.return_value = create_mock_process(stdout=GEMINI_JSON_RESPONSE)

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [{"cli": "gemini", "prompt": "Hello world test"}],
                "elicit": False,
            },
        )

        assert result.is_error is False
        assert result.data.succeeded == 1
        assert result.data.failed == 0
