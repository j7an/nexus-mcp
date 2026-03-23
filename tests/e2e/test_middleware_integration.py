# tests/e2e/test_middleware_integration.py
"""Integration test verifying the full middleware pipeline fires end-to-end.

Uses Client(mcp) in-process transport — no network. Verifies that all three
middleware (ErrorNormalization, Timing, RequestLogging) execute in the correct
order for a real tool call through the MCP protocol stack.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from tests.fixtures import make_agent_response


@pytest.mark.e2e
class TestMiddlewarePipeline:
    """Verify middleware fires for real MCP tool calls."""

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_full_pipeline_logs_for_successful_call(self, mock_factory, mcp_client, caplog):
        """A successful prompt call produces timing and request logs."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="hello")
        mock_factory.create.return_value = mock_runner

        with caplog.at_level(logging.DEBUG, logger="nexus_mcp.middleware"):
            await mcp_client.call_tool(
                "prompt",
                {"cli": "gemini", "prompt": "say hello"},
            )

        # RequestLoggingMiddleware entry + exit
        assert "Tool 'prompt' called" in caplog.text
        assert "cli=gemini" in caplog.text
        assert "Tool 'prompt' completed" in caplog.text

        # TimingMiddleware
        assert "completed in" in caplog.text
        assert "ms" in caplog.text

        # Prompt text not leaked in middleware logs
        middleware_text = " ".join(
            r.message for r in caplog.records if r.name == "nexus_mcp.middleware"
        )
        assert "say hello" not in middleware_text

    async def test_error_normalization_propagates_tool_error(self, mcp_client, caplog):
        """ToolError from batch_prompt propagates through the middleware stack.

        batch_prompt raises ValueError for max_concurrency < 1. FastMCP's tool
        runner wraps this into a ToolError before it reaches our middleware.
        ErrorNormalizationMiddleware passes ToolError through unchanged (the
        ``except ToolError: raise`` path). RequestLoggingMiddleware logs the
        failure, and TimingMiddleware records duration regardless.

        FastMCP's Client re-raises ToolError on the client side (raise_on_error=True
        by default). The middleware logs are captured before the exception propagates
        back to the test.
        """
        from fastmcp.exceptions import ToolError

        with (
            caplog.at_level(logging.DEBUG, logger="nexus_mcp.middleware"),
            pytest.raises(ToolError),
        ):
            await mcp_client.call_tool(
                "batch_prompt",
                {
                    "tasks": [{"cli": "gemini", "prompt": "hello"}],
                    "max_concurrency": 0,
                },
            )

        # RequestLoggingMiddleware logged entry and failure
        assert "Tool 'batch_prompt' called" in caplog.text
        assert "Tool 'batch_prompt' failed" in caplog.text
        assert "max_concurrency must be >= 1" in caplog.text

        # TimingMiddleware still logged duration
        assert "completed in" in caplog.text
