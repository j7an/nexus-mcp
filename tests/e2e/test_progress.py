# tests/e2e/test_progress.py
"""E2E tests for progress reporting through the full MCP protocol stack.

Tests verify that ctx.report_progress calls flow from runners through
server-side emitter factories to the MCP client, exercising the real
FastMCP DI injection and JSON-RPC dispatch pipeline.

Mock boundary: asyncio.create_subprocess_exec only.
"""

from unittest.mock import patch

import pytest

from tests.fixtures import CODEX_NDJSON_RESPONSE, create_mock_process


@pytest.mark.e2e
class TestSinglePromptProgress:
    """Verify single prompt reports runner-level progress."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_prompt_reports_step_progress(self, mock_exec, progress_mcp_client):
        """Single prompt tool call should report step-level progress."""
        mock_exec.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE, returncode=0)
        client, progress_events = progress_mcp_client

        result = await client.call_tool("prompt", {"cli": "codex", "prompt": "hello"})

        assert result is not None
        assert mock_exec.await_count == 1
        assert progress_events[0] == (1.0, 1.0, "Attempt 1/1")
        assert all("Task '" not in (message or "") for _, _, message in progress_events)


@pytest.mark.e2e
class TestBatchPromptProgress:
    """Verify batch prompt reports hierarchical progress."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_batch_reports_progress(self, mock_exec, progress_mcp_client):
        """Multi-task batch should complete with progress reporting active."""
        mock_exec.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE, returncode=0)
        client, progress_events = progress_mcp_client

        result = await client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "codex", "prompt": "task1", "label": "first"},
                    {"cli": "codex", "prompt": "task2", "label": "second"},
                ],
            },
        )

        assert result is not None
        assert mock_exec.await_count == 2
        assert all(total == 2.0 for _, total, _ in progress_events)
        messages = [message or "" for _, _, message in progress_events]
        assert any(message.startswith("Task 'first' (1/2):") for message in messages)
        assert any(message.startswith("Task 'second' (2/2):") for message in messages)

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_single_task_batch_uses_unwrapped_progress(self, mock_exec, progress_mcp_client):
        """Single-task batch should use unwrapped (passthrough) progress."""
        mock_exec.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE, returncode=0)
        client, progress_events = progress_mcp_client

        result = await client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "codex", "prompt": "solo task"},
                ],
            },
        )

        assert result is not None
        assert mock_exec.await_count == 1
        assert progress_events[0] == (1.0, 1.0, "Attempt 1/1")
        assert all("Task '" not in (message or "") for _, _, message in progress_events)
