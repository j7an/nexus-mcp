# src/nexus_mcp/server.py
"""FastMCP server with CLI agent tools.

Exposes two MCP tools:
- prompt_agent: Send a prompt to a CLI agent (background task, returns immediately)
- list_agents: Return list of supported agent names

Background task design: prompt_agent uses @mcp.tool(task=True) so it runs
asynchronously and returns a task ID immediately. This prevents MCP timeouts
for long operations like YOLO mode (2-5 minutes).

Testability: The tool functions are defined as plain async functions and then
registered with @mcp.tool so tests can import and call them directly without
going through the FunctionTool wrapper.
"""

from typing import Any

from fastmcp import FastMCP
from fastmcp.dependencies import Progress

from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.types import ExecutionMode, PromptRequest

mcp = FastMCP("nexus-mcp")


async def prompt_agent(
    agent: str,
    prompt: str,
    progress: Progress = Progress(),  # noqa: B008 -- FastMCP DI sentinel pattern
    context: dict[str, Any] | None = None,
    execution_mode: ExecutionMode = "default",
    model: str | None = None,
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
mcp.tool(task=True)(prompt_agent)
mcp.tool()(list_agents)
