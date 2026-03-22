# tests/unit/test_middleware.py
"""Tests for FastMCP middleware classes.

Mocking strategy: Mock call_next (the boundary). Each middleware is tested in
isolation with a mock MiddlewareContext and a mock call_next callable.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import mcp.types as mt
import pytest
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult


@dataclass(frozen=True)
class FakeMiddlewareContext:
    """Minimal stand-in for MiddlewareContext[CallToolRequestParams].

    MiddlewareContext is a frozen dataclass — we replicate the fields our
    middleware actually reads so tests don't depend on FastMCP internals.
    """

    message: mt.CallToolRequestParams
    method: str | None = "tools/call"
    source: str = "client"
    type: str = "request"
    timestamp: datetime | None = None
    fastmcp_context: object | None = None


def _make_context(
    tool_name: str = "prompt",
    arguments: dict | None = None,
) -> FakeMiddlewareContext:
    """Build a FakeMiddlewareContext for a tool call."""
    return FakeMiddlewareContext(
        message=mt.CallToolRequestParams(
            name=tool_name,
            arguments=arguments or {},
        ),
        timestamp=datetime.now(UTC),
    )


def _make_tool_result(text: str = "ok") -> ToolResult:
    """Build a minimal ToolResult for call_next return values."""
    return ToolResult(content=[mt.TextContent(type="text", text=text)])


class TestTimingMiddleware:
    """Tests for TimingMiddleware."""

    async def test_logs_duration_on_success(self, caplog):
        """Successful tool call logs duration at DEBUG level."""
        from nexus_mcp.middleware import TimingMiddleware

        mw = TimingMiddleware()
        ctx = _make_context("prompt")
        call_next = AsyncMock(return_value=_make_tool_result())

        with (
            patch("nexus_mcp.middleware.perf_counter", side_effect=[0.0, 1.5]),
            caplog.at_level(logging.DEBUG, logger="nexus_mcp.middleware"),
        ):
            result = await mw.on_call_tool(ctx, call_next)

        call_next.assert_awaited_once_with(ctx)
        assert result == call_next.return_value
        assert "Tool 'prompt' completed in 1500ms" in caplog.text

    async def test_logs_duration_on_failure(self, caplog):
        """Failed tool call still logs duration at DEBUG level."""
        from nexus_mcp.middleware import TimingMiddleware

        mw = TimingMiddleware()
        ctx = _make_context("batch_prompt")
        call_next = AsyncMock(side_effect=ToolError("boom"))

        with (
            patch("nexus_mcp.middleware.perf_counter", side_effect=[0.0, 2.0]),
            caplog.at_level(logging.DEBUG, logger="nexus_mcp.middleware"),
            pytest.raises(ToolError, match="boom"),
        ):
            await mw.on_call_tool(ctx, call_next)

        assert "Tool 'batch_prompt' completed in 2000ms" in caplog.text

    async def test_duration_is_positive(self, caplog):
        """Duration logged is always non-negative (sanity check via mock)."""
        from nexus_mcp.middleware import TimingMiddleware

        mw = TimingMiddleware()
        ctx = _make_context("get_preferences")
        call_next = AsyncMock(return_value=_make_tool_result())

        with (
            patch("nexus_mcp.middleware.perf_counter", side_effect=[100.0, 100.003]),
            caplog.at_level(logging.DEBUG, logger="nexus_mcp.middleware"),
        ):
            await mw.on_call_tool(ctx, call_next)

        assert "completed in 3ms" in caplog.text
