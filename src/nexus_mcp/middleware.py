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
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult

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
