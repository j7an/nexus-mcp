# src/nexus_mcp/middleware.py
"""FastMCP middleware for protocol-level observability and error normalization.

Three middleware classes, registered outermost → innermost:
1. ErrorNormalizationMiddleware — catches exceptions, maps to ToolError
2. TimingMiddleware — measures wall-clock duration of tool calls
3. RequestLoggingMiddleware — logs tool call entry/exit

All hook on_call_tool only. All use Python logging (server-side), not ctx
(MCP-client-side) — runner-level LogEmitter already handles client notifications.
"""

import logging
from time import perf_counter

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult
from pydantic import ValidationError

from nexus_mcp.exceptions import CLINotFoundError, UnsupportedAgentError

logger = logging.getLogger(__name__)


class TimingMiddleware(Middleware):
    """Measure wall-clock duration of every tool call.

    Logs at DEBUG level — opt in via log configuration.
    Uses time.perf_counter() for monotonic, high-resolution timing.
    Includes failed calls (duration logged in finally block).
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        start = perf_counter()
        try:
            return await call_next(context)
        finally:
            elapsed_ms = int((perf_counter() - start) * 1000)
            logger.debug("Tool '%s' completed in %dms", tool_name, elapsed_ms)


# Tools that use prompt text — these get arg summarization, not full dump.
_PROMPT_TOOLS = frozenset({"prompt", "batch_prompt"})


def _summarize_args(tool_name: str, arguments: dict[str, object] | None) -> str:
    """Build a concise arg summary for log messages.

    prompt/batch_prompt: log structured fields only (cli, model, execution_mode,
    task count) — never prompt text (large, potentially sensitive).
    Other tools: log full arguments (small config values).
    """
    if not arguments:
        return ""

    if tool_name not in _PROMPT_TOOLS:
        # Preferences tools — safe to log everything
        parts = [f"{k}={v}" for k, v in arguments.items() if v is not None]
        return f" [{', '.join(parts)}]" if parts else ""

    if tool_name == "batch_prompt":
        raw_tasks = arguments.get("tasks", [])
        task_count = len(raw_tasks) if isinstance(raw_tasks, list) else 0
        parts = [f"tasks={task_count}"]
        max_conc = arguments.get("max_concurrency")
        if max_conc is not None:
            parts.append(f"max_concurrency={max_conc}")
        return f" [{', '.join(parts)}]"

    # prompt — log structured fields, skip prompt text and context
    safe_keys = ("cli", "model", "execution_mode", "max_retries", "timeout")
    parts = [
        f"{k}={arguments[k]}" for k in safe_keys if k in arguments and arguments[k] is not None
    ]
    return f" [{', '.join(parts)}]" if parts else ""


class RequestLoggingMiddleware(Middleware):
    """Log tool call entry/exit at the MCP protocol layer.

    Logs at INFO level via Python logging (server-side observability).
    Summarizes args for prompt/batch_prompt (redacts prompt text).
    Logs full args for preferences tools (small config values).
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        tool_name = context.message.name
        args_summary = _summarize_args(tool_name, context.message.arguments)
        logger.info("Tool '%s' called%s", tool_name, args_summary)
        try:
            result = await call_next(context)
            logger.info("Tool '%s' completed", tool_name)
            return result
        except Exception as e:
            logger.info("Tool '%s' failed: %s: %s", tool_name, type(e).__name__, e)
            raise


class ErrorNormalizationMiddleware(Middleware):
    """Centralized exception-to-ToolError mapping.

    Catches exceptions escaping tool functions and converts them to
    consistent ToolError responses. ToolError itself passes through
    unchanged. Unknown exceptions are logged with full traceback.

    Registered outermost so it catches errors from all downstream
    middleware and the tool handler.
    """

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        try:
            return await call_next(context)
        except ToolError:
            raise
        except (CLINotFoundError, UnsupportedAgentError) as e:
            raise ToolError(str(e)) from e
        except ValidationError as e:
            raise ToolError(f"Invalid input: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error in tool '%s'", context.message.name)
            raise ToolError(f"Internal error: {type(e).__name__}: {e}") from e
