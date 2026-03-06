# src/nexus_mcp/server.py
"""FastMCP server with CLI agent tools.

Exposes three MCP tools:
- batch_prompt: Send multiple prompts to CLI agents in parallel (primary tool)
- prompt: Send a single prompt to a CLI agent, routes to batch_prompt
- list_agents: Return list of supported agent names

Background task design: both prompt and batch_prompt use @mcp.tool(task=True) so they
run asynchronously and return task IDs immediately. This prevents MCP timeouts for long
operations like YOLO mode (2-5 minutes).

Testability: The tool functions are defined as plain async functions and then
registered with @mcp.tool so tests can import and call them directly without
going through the FunctionTool wrapper.
"""

import asyncio
import logging
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.dependencies import Progress, ProgressLike
from fastmcp.exceptions import ToolError

from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.types import (
    DEFAULT_MAX_CONCURRENCY,
    AgentTask,
    AgentTaskResult,
    ExecutionMode,
    MultiPromptResponse,
)

mcp = FastMCP("nexus-mcp")
logger = logging.getLogger(__name__)


def _next_available_label(base: str, reserved: set[str]) -> str:
    """Return the first available label derived from base, avoiding reserved names.

    Returns base if available, otherwise base-2, base-3, etc.

    Args:
        base: Preferred label (typically agent name).
        reserved: Set of already-taken labels.

    Returns:
        base if not reserved, otherwise base-N for the lowest N >= 2 not in reserved.
    """
    if base not in reserved:
        return base
    n = 2
    while f"{base}-{n}" in reserved:
        n += 1
    return f"{base}-{n}"


def _assign_labels(tasks: list[AgentTask]) -> list[AgentTask]:
    """Assign unique labels to tasks, preserving explicit ones.

    Two-pass algorithm:
    1. Reserve all explicit labels
    2. Auto-assign from agent name with -N suffixes for collisions

    Args:
        tasks: List of AgentTask objects (may have label=None).

    Returns:
        New list of AgentTask objects with label set on every item.
        Input tasks are never mutated (model_copy() used throughout).
    """
    reserved = {t.label for t in tasks if t.label is not None}

    result: list[AgentTask] = []
    for task in tasks:
        if task.label is not None:
            result.append(task)
            continue

        label = _next_available_label(task.agent, reserved)
        reserved.add(label)
        result.append(task.model_copy(update={"label": label}))

    return result


async def batch_prompt(
    tasks: list[AgentTask],
    progress: ProgressLike = Progress(),  # noqa: B008
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ctx: Context | None = None,
) -> MultiPromptResponse:
    """Send multiple prompts to CLI agents in parallel (primary tool).

    Fans out tasks server-side with asyncio.gather and a semaphore, enabling
    true parallel agent execution within a single MCP call. Single-task usage
    is perfectly valid — use prompt for convenience when sending one task.

    Args:
        tasks: List of AgentTask objects, each with agent, prompt, and optional fields.
        progress: Progress tracker (auto-injected by FastMCP).
        max_concurrency: Max parallel agent invocations (default: 3).
        ctx: MCP context (auto-injected by FastMCP). None when called directly in tests.

    Returns:
        MultiPromptResponse with results for each task.
    """
    # Docket serializes/deserializes task arguments as JSON when task=True,
    # converting AgentTask objects to plain dicts. Reconstruct them here.
    tasks = [AgentTask(**t) if isinstance(t, dict) else t for t in tasks]

    labelled = _assign_labels(tasks)
    semaphore = asyncio.Semaphore(max_concurrency)

    await progress.set_total(len(labelled))
    if ctx:
        await ctx.info(f"Starting batch of {len(labelled)} tasks (concurrency={max_concurrency})")

    async def _run_single(task: AgentTask) -> AgentTaskResult:
        async with semaphore:
            try:
                request = task.to_request()
                runner = RunnerFactory.create(task.agent)
                response = await runner.run(request)
                return AgentTaskResult(label=task.label, output=response.output)  # type: ignore[arg-type]
            except Exception as e:
                logger.exception("Task %r failed: %s", task.label, e)
                return AgentTaskResult(label=task.label, error=str(e), error_type=type(e).__name__)  # type: ignore[arg-type]
            finally:
                await progress.increment(1)

    results = await asyncio.gather(*[_run_single(t) for t in labelled])
    response = MultiPromptResponse(results=list(results))
    if ctx:
        await ctx.info(f"Batch complete: {response.succeeded}/{response.total} succeeded")
    return response


async def prompt(
    agent: str,
    prompt: str,
    progress: ProgressLike = Progress(),  # noqa: B008
    context: dict[str, Any] | None = None,
    execution_mode: ExecutionMode = "default",
    model: str | None = None,
    max_retries: int | None = None,
    ctx: Context | None = None,
) -> str:
    """Send a prompt to a CLI agent as a background task.

    Returns immediately with a task ID. Client polls for results.
    This prevents timeouts for long operations (YOLO mode: 2-5 minutes).

    Args:
        agent: Agent name (e.g., "gemini")
        prompt: Prompt text to send to the agent
        progress: Progress tracker (auto-injected by FastMCP)
        context: Optional context metadata
        execution_mode: 'default' (safe), 'sandbox', or 'yolo'
        model: Optional model name (uses CLI default if not specified)
        max_retries: Max retry attempts for transient errors (None uses env default)
        ctx: MCP context (auto-injected by FastMCP). None when called directly in tests.

    Returns:
        Agent's response text
    """
    task = AgentTask(
        agent=agent,
        prompt=prompt,
        context=context or {},
        execution_mode=execution_mode,
        model=model,
        max_retries=max_retries,
    )
    result = await batch_prompt(tasks=[task], progress=progress, ctx=ctx)
    task_result = result.results[0]
    if task_result.error:
        raise ToolError(task_result.formatted_error)
    return task_result.output  # type: ignore[return-value]


def list_agents() -> list[str]:
    """Return list of supported agent names.

    Returns:
        List of agent names that can be used with prompt or batch_prompt.
    """
    return RunnerFactory.list_agents()


# Register functions as MCP tools after definition so tests can import
# and call the raw functions directly (not the FunctionTool wrappers).
mcp.tool(task=True)(batch_prompt)
mcp.tool(task=True)(prompt)
mcp.tool()(list_agents)
