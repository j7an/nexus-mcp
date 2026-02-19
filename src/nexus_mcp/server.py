# src/nexus_mcp/server.py
"""FastMCP server with CLI agent tools.

Exposes three MCP tools:
- batch_prompt: Send multiple prompts to CLI agents in parallel (primary tool)
- prompt_agent: Send a prompt to a CLI agent (background task) — DEPRECATED, use batch_prompt
- list_agents: Return list of supported agent names

Background task design: both prompt_agent and batch_prompt use @mcp.tool(task=True) so they
run asynchronously and return task IDs immediately. This prevents MCP timeouts for long
operations like YOLO mode (2-5 minutes).

Testability: The tool functions are defined as plain async functions and then
registered with @mcp.tool so tests can import and call them directly without
going through the FunctionTool wrapper.
"""

import asyncio
from typing import Any

from fastmcp import FastMCP
from fastmcp.dependencies import Progress

from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.types import (
    DEFAULT_MAX_CONCURRENCY,
    AgentTask,
    AgentTaskResult,
    ExecutionMode,
    MultiPromptResponse,
    PromptRequest,
)

mcp = FastMCP("nexus-mcp")


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

        base = task.agent
        if base not in reserved:
            reserved.add(base)
            result.append(task.model_copy(update={"label": base}))
        else:
            n = 2
            while f"{base}-{n}" in reserved:
                n += 1
            label = f"{base}-{n}"
            reserved.add(label)
            result.append(task.model_copy(update={"label": label}))

    return result


async def batch_prompt(
    tasks: list[AgentTask],
    progress: Progress = Progress(),  # noqa: B008 -- FastMCP DI sentinel pattern
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> str:
    """Send multiple prompts to CLI agents in parallel (primary tool).

    Fans out tasks server-side with asyncio.gather and a semaphore, enabling
    true parallel agent execution within a single MCP call. Single-task usage
    is perfectly valid — batch_prompt replaces prompt_agent (now deprecated).

    Args:
        tasks: List of AgentTask objects, each with agent, prompt, and optional fields.
        progress: Progress tracker (auto-injected by FastMCP).
        max_concurrency: Max parallel agent invocations (default: 3).

    Returns:
        JSON string containing MultiPromptResponse with results for each task.
    """
    labelled = _assign_labels(tasks)
    semaphore = asyncio.Semaphore(max_concurrency)

    await progress.set_total(len(labelled))

    async def _run_single(task: AgentTask) -> AgentTaskResult:
        async with semaphore:
            try:
                request = PromptRequest(
                    agent=task.agent,
                    prompt=task.prompt,
                    context=task.context,
                    execution_mode=task.execution_mode,
                    model=task.model,
                )
                runner = RunnerFactory.create(task.agent)
                response = await runner.run(request)
                await progress.increment(1)
                return AgentTaskResult(label=task.label, output=response.output)  # type: ignore[arg-type]
            except Exception as e:
                await progress.increment(1)
                return AgentTaskResult(label=task.label, error=str(e))  # type: ignore[arg-type]

    results = await asyncio.gather(*[_run_single(t) for t in labelled])
    return MultiPromptResponse(results=list(results)).model_dump_json()


async def prompt_agent(
    agent: str,
    prompt: str,
    progress: Progress = Progress(),  # noqa: B008 -- FastMCP DI sentinel pattern
    context: dict[str, Any] | None = None,
    execution_mode: ExecutionMode = "default",
    model: str | None = None,
) -> str:
    """Send a prompt to a CLI agent as a background task.

    .. deprecated::
        Use ``batch_prompt`` instead. ``batch_prompt`` handles both single and
        multi-task cases and is the primary tool going forward. ``prompt_agent``
        remains functional for backward compatibility during the transition.

    Returns immediately with a task ID. Client polls for results.
    This prevents timeouts for long operations (YOLO mode: 2-5 minutes).

    Args:
        agent: Agent name (e.g., "gemini")
        prompt: Prompt text to send to the agent
        progress: Progress tracker (auto-injected by FastMCP)
        context: Optional context metadata
        execution_mode: 'default' (safe), 'sandbox', or 'yolo'
        model: Optional model name (uses CLI default if not specified)

    Returns:
        Agent's response text
    """
    await progress.set_total(100)
    await progress.increment(10)

    request = PromptRequest(
        agent=agent,
        prompt=prompt,
        context=context or {},
        execution_mode=execution_mode,
        model=model,
    )

    await progress.increment(20)

    runner = RunnerFactory.create(agent)

    await progress.increment(30)

    response = await runner.run(request)

    await progress.increment(40)

    return response.output


def list_agents() -> list[str]:
    """Return list of supported agent names.

    Returns:
        List of agent names that can be used with prompt_agent.
    """
    return RunnerFactory.list_agents()


# Register functions as MCP tools after definition so tests can import
# and call the raw functions directly (not the FunctionTool wrappers).
mcp.tool(task=True)(batch_prompt)
mcp.tool(task=True)(prompt_agent)
mcp.tool()(list_agents)
