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
from pydantic import ValidationError

from nexus_mcp.exceptions import CLINotFoundError, UnsupportedAgentError
from nexus_mcp.middleware import (
    ErrorNormalizationMiddleware,
    RequestLoggingMiddleware,
    TimingMiddleware,
)


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
        mw = TimingMiddleware()
        ctx = _make_context("get_preferences")
        call_next = AsyncMock(return_value=_make_tool_result())

        with (
            patch("nexus_mcp.middleware.perf_counter", side_effect=[100.0, 100.003]),
            caplog.at_level(logging.DEBUG, logger="nexus_mcp.middleware"),
        ):
            await mw.on_call_tool(ctx, call_next)

        assert "completed in 3ms" in caplog.text


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""

    async def test_logs_entry_and_exit(self, caplog):
        """Successful tool call logs entry with args and exit."""
        mw = RequestLoggingMiddleware()
        ctx = _make_context("prompt", {"cli": "gemini", "prompt": "tell me a joke"})
        call_next = AsyncMock(return_value=_make_tool_result())

        with caplog.at_level(logging.INFO, logger="nexus_mcp.middleware"):
            result = await mw.on_call_tool(ctx, call_next)

        assert result == call_next.return_value
        assert "Tool 'prompt' called" in caplog.text
        assert "cli=gemini" in caplog.text
        assert "Tool 'prompt' completed" in caplog.text

    async def test_logs_failure(self, caplog):
        """Failed tool call logs entry and failure with exception type."""
        mw = RequestLoggingMiddleware()
        ctx = _make_context("prompt", {"cli": "codex", "prompt": "hello"})
        call_next = AsyncMock(side_effect=ToolError("codex not found"))

        with (
            caplog.at_level(logging.INFO, logger="nexus_mcp.middleware"),
            pytest.raises(ToolError),
        ):
            await mw.on_call_tool(ctx, call_next)

        assert "Tool 'prompt' called" in caplog.text
        assert "Tool 'prompt' failed" in caplog.text
        assert "codex not found" in caplog.text

    async def test_redacts_prompt_text(self, caplog):
        """Prompt text must NOT appear in log messages (privacy/size)."""
        mw = RequestLoggingMiddleware()
        secret_prompt = "tell me the nuclear launch codes"
        ctx = _make_context(
            "prompt",
            {"cli": "gemini", "prompt": secret_prompt, "execution_mode": "default"},
        )
        call_next = AsyncMock(return_value=_make_tool_result())

        with caplog.at_level(logging.DEBUG, logger="nexus_mcp.middleware"):
            await mw.on_call_tool(ctx, call_next)

        assert secret_prompt not in caplog.text

    async def test_logs_batch_prompt_task_count(self, caplog):
        """batch_prompt logs task count, not individual task details."""
        mw = RequestLoggingMiddleware()
        ctx = _make_context(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "gemini", "prompt": "secret1"},
                    {"cli": "codex", "prompt": "secret2"},
                ],
            },
        )
        call_next = AsyncMock(return_value=_make_tool_result())

        with caplog.at_level(logging.INFO, logger="nexus_mcp.middleware"):
            await mw.on_call_tool(ctx, call_next)

        assert "tasks=2" in caplog.text
        assert "secret1" not in caplog.text
        assert "secret2" not in caplog.text

    async def test_logs_full_args_for_preferences(self, caplog):
        """Preferences tools log full args (small config values, no secrets)."""
        mw = RequestLoggingMiddleware()
        ctx = _make_context(
            "set_preferences",
            {"execution_mode": "yolo", "model": "gemini-2.5-flash"},
        )
        call_next = AsyncMock(return_value=_make_tool_result())

        with caplog.at_level(logging.INFO, logger="nexus_mcp.middleware"):
            await mw.on_call_tool(ctx, call_next)

        assert "execution_mode" in caplog.text
        assert "yolo" in caplog.text
        assert "gemini-2.5-flash" in caplog.text


class TestErrorNormalizationMiddleware:
    """Tests for ErrorNormalizationMiddleware."""

    async def test_successful_call_passes_through(self):
        """Successful tool calls are returned unmodified."""
        mw = ErrorNormalizationMiddleware()
        ctx = _make_context("prompt")
        expected = _make_tool_result("success")
        call_next = AsyncMock(return_value=expected)

        result = await mw.on_call_tool(ctx, call_next)

        assert result is expected

    async def test_tool_error_passes_through(self):
        """ToolError is re-raised unchanged (already normalized)."""
        mw = ErrorNormalizationMiddleware()
        ctx = _make_context("prompt")
        call_next = AsyncMock(side_effect=ToolError("already clean"))

        with pytest.raises(ToolError, match="already clean"):
            await mw.on_call_tool(ctx, call_next)

    async def test_cli_not_found_becomes_tool_error(self):
        """CLINotFoundError is mapped to ToolError with clean message."""
        mw = ErrorNormalizationMiddleware()
        ctx = _make_context("prompt")
        call_next = AsyncMock(side_effect=CLINotFoundError("codex"))

        with pytest.raises(ToolError, match="CLI not found in PATH: codex"):
            await mw.on_call_tool(ctx, call_next)

    async def test_unsupported_agent_becomes_tool_error(self):
        """UnsupportedAgentError is mapped to ToolError."""
        mw = ErrorNormalizationMiddleware()
        ctx = _make_context("prompt")
        call_next = AsyncMock(side_effect=UnsupportedAgentError("unknown_cli"))

        with pytest.raises(ToolError, match="Unsupported agent: unknown_cli"):
            await mw.on_call_tool(ctx, call_next)

    async def test_validation_error_becomes_tool_error(self):
        """Pydantic ValidationError is mapped to ToolError with 'Invalid input' prefix."""
        from nexus_mcp.types import SessionPreferences

        mw = ErrorNormalizationMiddleware()
        ctx = _make_context("set_preferences")

        # Capture a real ValidationError from pydantic
        with pytest.raises(ValidationError) as exc_info:
            SessionPreferences(max_retries=-1)  # ge=1 constraint
        call_next = AsyncMock(side_effect=exc_info.value)

        with pytest.raises(ToolError, match="Invalid input"):
            await mw.on_call_tool(ctx, call_next)

    async def test_unexpected_exception_becomes_internal_error(self, caplog):
        """Unknown exceptions become 'Internal error' ToolError and are logged."""
        mw = ErrorNormalizationMiddleware()
        ctx = _make_context("prompt")
        call_next = AsyncMock(side_effect=RuntimeError("something broke"))

        with (
            caplog.at_level(logging.ERROR, logger="nexus_mcp.middleware"),
            pytest.raises(ToolError, match="Internal error: RuntimeError: something broke"),
        ):
            await mw.on_call_tool(ctx, call_next)

        assert "Unexpected error in tool 'prompt'" in caplog.text
