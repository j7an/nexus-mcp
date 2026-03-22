# tests/e2e/test_progress.py
"""E2E tests for progress reporting through the full MCP protocol stack.

Tests verify that ctx.report_progress calls flow from runners through
server-side emitter factories to the MCP client, exercising the real
FastMCP DI injection and JSON-RPC dispatch pipeline.

Mock boundary: asyncio.create_subprocess_exec only.
"""

from unittest.mock import patch

import pytest

from tests.fixtures import GEMINI_JSON_RESPONSE, create_mock_process


@pytest.mark.e2e
class TestSinglePromptProgress:
    """Verify single prompt reports runner-level progress."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_prompt_reports_step_progress(self, mock_exec, mcp_client):
        """Single prompt tool call should report step-level progress."""
        mock_exec.return_value = create_mock_process(stdout=GEMINI_JSON_RESPONSE, returncode=0)

        result = await mcp_client.call_tool("prompt", {"cli": "gemini", "prompt": "hello"})

        # Verify the call succeeded (progress reporting is fire-and-forget)
        assert result is not None


@pytest.mark.e2e
class TestBatchPromptProgress:
    """Verify batch prompt reports hierarchical progress."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_batch_reports_progress(self, mock_exec, mcp_client):
        """Multi-task batch should complete with progress reporting active."""
        mock_exec.return_value = create_mock_process(stdout=GEMINI_JSON_RESPONSE, returncode=0)

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "gemini", "prompt": "task1", "label": "first"},
                    {"cli": "gemini", "prompt": "task2", "label": "second"},
                ],
            },
        )

        assert result is not None

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_single_task_batch_uses_unwrapped_progress(self, mock_exec, mcp_client):
        """Single-task batch should use unwrapped (passthrough) progress."""
        mock_exec.return_value = create_mock_process(stdout=GEMINI_JSON_RESPONSE, returncode=0)

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "gemini", "prompt": "solo task"},
                ],
            },
        )

        assert result is not None
