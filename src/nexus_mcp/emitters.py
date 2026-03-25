"""MCP emitter adapters for nexus-mcp.

Bridges FastMCP's Context logging/progress APIs to the LogEmitter and
ProgressEmitter protocols consumed by runners.
"""

import logging

from fastmcp import Context

from nexus_mcp.types import LogEmitter, LogLevel, ProgressEmitter

logger = logging.getLogger(__name__)


def make_mcp_emitter(ctx: Context) -> LogEmitter:
    """Create a LogEmitter that sends to both MCP client and Python logger.

    Error-level messages use logger.error(exc_info=True) to preserve tracebacks
    on stderr for server operators, while MCP clients get a clean message.
    """
    _ctx_methods = {
        "debug": ctx.debug,
        "info": ctx.info,
        "warning": ctx.warning,
        "error": ctx.error,
    }

    async def _emit(level: LogLevel, message: str) -> None:
        await _ctx_methods[level](message)
        if level == "error":
            logger.error(message, exc_info=True)
        else:
            getattr(logger, level)(message)

    return _emit


def make_progress_emitter(
    ctx: Context,
    *,
    task_idx: int | None = None,
    task_count: int | None = None,
    label: str | None = None,
) -> ProgressEmitter:
    """Create a ProgressEmitter that bridges to ctx.report_progress.

    Single-task mode (default): runner's progress/total pass through directly.
    Batch mode (task_idx + task_count set): wraps with task-level counters.
    """
    if task_idx is not None and task_count is not None:
        _idx, _count, _label = task_idx, task_count, label

        async def _report(progress: float, total: float, message: str) -> None:
            await ctx.report_progress(
                progress=_idx,
                total=_count,
                message=f"Task '{_label}' ({_idx}/{_count}): {message}",
            )
    else:

        async def _report(progress: float, total: float, message: str) -> None:
            await ctx.report_progress(progress=progress, total=total, message=message)

    return _report
