# src/nexus_mcp/server.py
"""FastMCP server with CLI agent tools.

Exposes three MCP tools:
- batch_prompt: Send multiple prompts to CLI agents in parallel (primary tool)
- prompt: Send a single prompt to a CLI agent, routes to batch_prompt
- list_runners: Return metadata for all registered CLI runners

Background task design: both prompt and batch_prompt use @mcp.tool(task=True) so they
run asynchronously and return task IDs immediately. This prevents MCP timeouts for long
operations like YOLO mode (2-5 minutes).

Testability: The tool functions are defined as plain async functions and then
registered with @mcp.tool so tests can import and call them directly without
going through the FunctionTool wrapper.
"""

import asyncio
import json
import logging
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from nexus_mcp.cli_detector import CLIInfo, detect_cli
from nexus_mcp.config import RunnerConfig, get_agent_env, get_tool_timeout, load_runner_config
from nexus_mcp.exceptions import CLINotFoundError
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.types import (
    DEFAULT_MAX_CONCURRENCY,
    AgentTask,
    AgentTaskResult,
    ExecutionMode,
    MultiPromptResponse,
    RunnerInfo,
    SessionPreferences,
)

mcp = FastMCP("nexus-mcp")
logger = logging.getLogger(__name__)

_PREFERENCES_KEY = "nexus:preferences"
_runner_config: dict[str, RunnerConfig] = load_runner_config()


async def _get_session_preferences(ctx: Context | None) -> SessionPreferences:
    """Read session preferences from ctx state, returning defaults when unset or ctx is None."""
    if ctx is None:
        return SessionPreferences()
    raw = await ctx.get_state(_PREFERENCES_KEY)
    if raw is None:
        return SessionPreferences()
    try:
        return SessionPreferences(**raw)  # reconstruct from dict after JSON round-trip
    except (ValidationError, TypeError) as e:
        raise ToolError(f"Session preferences are corrupted and cannot be loaded: {e}") from e


def _apply_preferences(task: AgentTask, prefs: SessionPreferences) -> AgentTask:
    """Fill in None fields on a task from session preferences.

    Returns the same task object if no updates are needed, otherwise a model_copy.
    Primary resolution happens in prompt(); this is the safety net for batch_prompt tasks.
    """
    updates: dict[str, Any] = {}
    if task.execution_mode is None:
        updates["execution_mode"] = prefs.execution_mode or "default"
    if task.model is None and prefs.model is not None:
        updates["model"] = prefs.model
    if task.max_retries is None and prefs.max_retries is not None:
        updates["max_retries"] = prefs.max_retries
    if task.output_limit is None and prefs.output_limit is not None:
        updates["output_limit"] = prefs.output_limit
    if task.timeout is None and prefs.timeout is not None:
        updates["timeout"] = prefs.timeout
    if updates:
        return task.model_copy(update=updates)
    return task


def _next_available_label(base: str, reserved: set[str]) -> str:
    """Return the first available label derived from base, avoiding reserved names.

    Returns base if available, otherwise base-2, base-3, etc.

    Args:
        base: Preferred label (typically cli name).
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
    2. Auto-assign from cli name with -N suffixes for collisions

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

        label = _next_available_label(task.cli, reserved)
        reserved.add(label)
        result.append(task.model_copy(update={"label": label}))

    return result


async def batch_prompt(
    tasks: list[AgentTask],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ctx: Context | None = None,
) -> MultiPromptResponse:
    """Send multiple prompts to CLI runners in parallel (primary tool).

    Fans out tasks server-side with asyncio.gather and a semaphore, enabling
    true parallel runner execution within a single MCP call. Single-task usage
    is perfectly valid — use prompt for convenience when sending one task.

    Args:
        tasks: List of AgentTask objects, each with cli, prompt, and optional fields.
        max_concurrency: Max parallel runner invocations (default: 3).
        ctx: MCP context (auto-injected by FastMCP). None when called directly in tests.

    Returns:
        MultiPromptResponse with results for each task.
    """
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

    # Docket serializes/deserializes task arguments as JSON when task=True,
    # converting AgentTask objects to plain dicts. Reconstruct them here.
    tasks = [AgentTask(**t) if isinstance(t, dict) else t for t in tasks]

    # Resolve session preferences before dispatching — workers in a separate process
    # (Redis-backed Docket) cannot access the foreground session's MemoryStore.
    prefs = await _get_session_preferences(ctx)
    tasks = [_apply_preferences(t, prefs) for t in tasks]

    labelled = _assign_labels(tasks)
    semaphore = asyncio.Semaphore(max_concurrency)
    completed = 0

    if ctx:
        await ctx.info(f"Starting batch of {len(labelled)} tasks (concurrency={max_concurrency})")

    async def _run_single(task: AgentTask) -> AgentTaskResult:
        nonlocal completed
        async with semaphore:
            try:
                request = task.to_request()
                runner = RunnerFactory.create(task.cli)
                response = await runner.run(request)
                return AgentTaskResult(label=task.label, output=response.output)  # type: ignore[arg-type]
            except Exception as e:
                logger.exception("Task %r failed: %s", task.label, e)
                return AgentTaskResult(label=task.label, error=str(e), error_type=type(e).__name__)  # type: ignore[arg-type]
            finally:
                completed += 1
                if ctx:
                    await ctx.report_progress(
                        progress=completed,
                        total=len(labelled),
                        message=f"Completed task {completed}/{len(labelled)}: {task.label}",
                    )

    results = await asyncio.gather(*[_run_single(t) for t in labelled])
    response = MultiPromptResponse(results=list(results))
    if ctx:
        await ctx.info(f"Batch complete: {response.succeeded}/{response.total} succeeded")
    return response


async def prompt(
    cli: str,
    prompt: str,
    context: dict[str, Any] | None = None,
    execution_mode: ExecutionMode | None = None,
    model: str | None = None,
    max_retries: int | None = None,
    output_limit: int | None = None,
    timeout: int | None = None,
    ctx: Context | None = None,
) -> str:
    """Send a prompt to a CLI runner as a background task.

    Returns immediately with a task ID. Client polls for results.
    This prevents timeouts for long operations (YOLO mode: 2-5 minutes).

    Args:
        cli: CLI runner name (e.g., "gemini")
        prompt: Prompt text to send to the runner
        context: Optional context metadata
        execution_mode: 'default' (safe) or 'yolo'. None inherits session preference.
        model: Optional model name. None inherits session preference or uses CLI default.
        max_retries: Max retry attempts for transient errors (None inherits session preference).
        output_limit: Max output bytes (None inherits session preference or uses env default).
        timeout: Subprocess timeout seconds (None inherits session preference or uses env default).
        ctx: MCP context (auto-injected by FastMCP). None when called directly in tests.

    Returns:
        Runner's response text
    """
    # Resolve session preferences here (foreground) so concrete values reach the Docket worker.
    prefs = await _get_session_preferences(ctx)
    session_mode = prefs.execution_mode or "default"
    resolved_mode = execution_mode if execution_mode is not None else session_mode
    resolved_model = model if model is not None else prefs.model
    resolved_max_retries = max_retries if max_retries is not None else prefs.max_retries
    resolved_output_limit = output_limit if output_limit is not None else prefs.output_limit
    resolved_timeout = timeout if timeout is not None else prefs.timeout

    task = AgentTask(
        cli=cli,
        prompt=prompt,
        context=context or {},
        execution_mode=resolved_mode,
        model=resolved_model,
        max_retries=resolved_max_retries,
        output_limit=resolved_output_limit,
        timeout=resolved_timeout,
    )
    result = await batch_prompt(tasks=[task], ctx=ctx)
    task_result = result.results[0]
    if task_result.error:
        raise ToolError(task_result.formatted_error)
    assert task_result.output is not None  # guaranteed: error is None, so output was set
    return task_result.output


def list_runners() -> list[RunnerInfo]:
    """Return metadata for all registered CLI runners.

    Returns:
        Sorted list of RunnerInfo with provider, models, availability,
        default model, and supported execution modes per runner.
    """
    result: list[RunnerInfo] = []
    for name in RunnerFactory.list_clis():
        cli_info = detect_cli(name)
        config = _runner_config.get(name)
        runner_cls = RunnerFactory.get_runner_class(name)

        if cli_info.found:
            try:
                instance = RunnerFactory.create(name)
                default_model = instance.default_model
            except CLINotFoundError:
                logger.warning("CLI '%s' disappeared between detection and creation", name)
                cli_info = CLIInfo(found=False)
                default_model = get_agent_env(name, "MODEL")
        else:
            default_model = get_agent_env(name, "MODEL")

        result.append(
            RunnerInfo(
                name=name,
                type=config.type if config else "cli",
                provider=config.provider if config else None,
                models=config.models if config else (),
                available=cli_info.found,
                default_model=default_model,
                execution_modes=runner_cls._SUPPORTED_MODES,
            )
        )
    return result


async def set_preferences(
    execution_mode: ExecutionMode | None = None,
    model: str | None = None,
    max_retries: int | None = None,
    output_limit: int | None = None,
    timeout: int | None = None,
    clear_execution_mode: bool = False,
    clear_model: bool = False,
    clear_max_retries: bool = False,
    clear_output_limit: bool = False,
    clear_timeout: bool = False,
    ctx: Context | None = None,
) -> str:
    """Set session-scoped preferences that apply to subsequent prompt/batch_prompt calls.

    Preferences persist for the duration of the MCP session. Call again to update,
    or use clear_preferences to reset all fields at once.

    To clear a single field while keeping others, pass the corresponding clear_* flag:
        set_preferences(clear_model=True)  # clears model, keeps execution_mode

    Args:
        execution_mode: Default execution mode for this session ('default' or 'yolo').
            None retains the current session value (use clear_execution_mode=True to reset).
        model: Default model name for this session (e.g. 'gemini-2.5-flash').
            None retains the current session value (use clear_model=True to reset).
        max_retries: Default max retry attempts for transient errors.
            None retains the current session value (use clear_max_retries=True to reset).
        output_limit: Default max output bytes per response.
            None retains the current session value (use clear_output_limit=True to reset).
        timeout: Default subprocess timeout in seconds.
            None retains the current session value (use clear_timeout=True to reset).
        clear_execution_mode: If True, resets execution_mode to None regardless of the
            execution_mode argument.
        clear_model: If True, resets model to None regardless of the model argument.
        clear_max_retries: If True, resets max_retries to None regardless of the argument.
        clear_output_limit: If True, resets output_limit to None regardless of the argument.
        clear_timeout: If True, resets timeout to None regardless of the argument.
        ctx: MCP context (auto-injected by FastMCP).

    Returns:
        Confirmation string with the active preferences as JSON.
    """
    if ctx is None:
        raise ToolError("set_preferences requires an active session context")
    existing = await _get_session_preferences(ctx)

    new_execution_mode: ExecutionMode | None
    if clear_execution_mode:
        new_execution_mode = None
    elif execution_mode is not None:
        new_execution_mode = execution_mode
    else:
        new_execution_mode = existing.execution_mode

    new_model: str | None
    if clear_model:
        new_model = None
    elif model is not None:
        new_model = model
    else:
        new_model = existing.model

    new_max_retries: int | None
    if clear_max_retries:
        new_max_retries = None
    elif max_retries is not None:
        new_max_retries = max_retries
    else:
        new_max_retries = existing.max_retries

    new_output_limit: int | None
    if clear_output_limit:
        new_output_limit = None
    elif output_limit is not None:
        new_output_limit = output_limit
    else:
        new_output_limit = existing.output_limit

    new_timeout: int | None
    if clear_timeout:
        new_timeout = None
    elif timeout is not None:
        new_timeout = timeout
    else:
        new_timeout = existing.timeout

    merged = SessionPreferences(
        execution_mode=new_execution_mode,
        model=new_model,
        max_retries=new_max_retries,
        output_limit=new_output_limit,
        timeout=new_timeout,
    )
    await ctx.set_state(_PREFERENCES_KEY, merged.model_dump())
    return f"Preferences set: {json.dumps(merged.model_dump())}"


async def get_preferences(ctx: Context | None = None) -> dict[str, Any]:
    """Return the current session preferences.

    Returns:
        Dict with 'execution_mode', 'model', 'max_retries', 'output_limit', and 'timeout'
        keys (None when unset).
    """
    if ctx is None:
        raise ToolError("get_preferences requires an active session context")
    prefs = await _get_session_preferences(ctx)
    return prefs.model_dump()


async def clear_preferences(ctx: Context | None = None) -> str:
    """Clear all session preferences, reverting to per-call defaults.

    Returns:
        Confirmation string.
    """
    if ctx is None:
        raise ToolError("clear_preferences requires an active session context")
    await ctx.delete_state(_PREFERENCES_KEY)
    return "Preferences cleared"


# Register functions as MCP tools after definition so tests can import
# and call the raw functions directly (not the FunctionTool wrappers).
# timeout wraps synchronous calls with anyio.fail_after(); when a client
# passes task=True, the call goes through Docket instead (subprocess-level
# timeout applies). Both prompt and batch_prompt support either path.
_tool_timeout = get_tool_timeout()
mcp.tool(task=True, timeout=_tool_timeout)(batch_prompt)
mcp.tool(task=True, timeout=_tool_timeout)(prompt)
mcp.tool()(list_runners)
mcp.tool()(set_preferences)
mcp.tool()(get_preferences)
mcp.tool()(clear_preferences)
