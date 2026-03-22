# src/nexus_mcp/server.py
"""FastMCP server with CLI agent tools.

Exposes five MCP tools:
- batch_prompt: Send multiple prompts to CLI agents in parallel (primary tool)
- prompt: Send a single prompt to a CLI agent, routes to batch_prompt
- set_preferences: Set session defaults (execution mode, model, retries, etc.)
- get_preferences: Retrieve current session preferences
- clear_preferences: Reset all session preferences

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
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field, ValidationError

from nexus_mcp.cli_detector import detect_cli
from nexus_mcp.config import get_runner_defaults, get_runner_models, get_tool_timeout
from nexus_mcp.middleware import (
    ErrorNormalizationMiddleware,
    RequestLoggingMiddleware,
    TimingMiddleware,
)
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.types import (
    DEFAULT_MAX_CONCURRENCY,
    AgentTask,
    AgentTaskResult,
    ExecutionMode,
    LogEmitter,
    LogLevel,
    MultiPromptResponse,
    SessionPreferences,
)


def build_server_instructions() -> str:
    """Generate markdown instructions describing available CLI runners.

    Called once at module load time. The instructions string is passed to
    FastMCP's constructor so MCP clients receive runner metadata on connection
    without needing a separate tool call.
    """
    lines = ["# nexus-mcp — CLI Agent Router", ""]
    lines.append("## Available Runners")
    lines.append("")

    for name in RunnerFactory.list_clis():
        cli_info = detect_cli(name)
        defaults = get_runner_defaults(name)
        runner_cls = RunnerFactory.get_runner_class(name)
        models = get_runner_models(name)

        status = "installed" if cli_info.found else "not found"
        lines.append(f"### {name} ({status})")

        if models:
            lines.append(f"- Models: {', '.join(models)}")
        if defaults.model:
            lines.append(f"- Default model: {defaults.model}")

        modes = ", ".join(runner_cls._SUPPORTED_MODES)
        lines.append(f"- Execution modes: {modes}")
        lines.append(f"- Default timeout: {defaults.timeout}s")
        lines.append("")

    return "\n".join(lines)


def _inject_cli_enum() -> None:
    """Inject runtime CLI names as JSON schema enum on cli parameters.

    Patches:
    1. prompt() function's ``cli`` parameter annotation — adds enum to schema
    2. AgentTask model's ``cli`` field — via json_schema_extra callable + model_rebuild()

    Called once before tool registration. The enum is schema-only — runtime
    validation stays with RunnerFactory.create() raising UnsupportedAgentError.
    """
    cli_names: list[Any] = list(RunnerFactory.list_clis())

    # 1. Patch prompt() cli parameter annotation
    prompt.__annotations__["cli"] = Annotated[str, Field(json_schema_extra={"enum": cli_names})]

    # 2. Patch AgentTask.cli field schema
    original_extra = AgentTask.model_config.get("json_schema_extra")

    def _add_cli_enum(schema: dict[str, Any]) -> None:
        if original_extra is not None:
            if callable(original_extra):
                original_extra(schema)  # type: ignore[call-arg]
            elif isinstance(original_extra, dict):
                schema.update(original_extra)
        # Inject enum into the cli property
        props = schema.get("properties", {})
        if "cli" in props:
            props["cli"]["enum"] = cli_names

    AgentTask.model_config["json_schema_extra"] = _add_cli_enum
    AgentTask.model_rebuild(force=True)


mcp = FastMCP("nexus-mcp", instructions=build_server_instructions())

# Middleware executes outermost → innermost on request, reverse on response.
# Order: ErrorNormalization (catch all) → Timing (measure) → RequestLogging (log entry/exit)
mcp.add_middleware(ErrorNormalizationMiddleware())
mcp.add_middleware(TimingMiddleware())
mcp.add_middleware(RequestLoggingMiddleware())

logger = logging.getLogger(__name__)


def _make_mcp_emitter(ctx: Context) -> LogEmitter:
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


_PREFERENCES_KEY = "nexus:preferences"


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
    if task.retry_base_delay is None and prefs.retry_base_delay is not None:
        updates["retry_base_delay"] = prefs.retry_base_delay
    if task.retry_max_delay is None and prefs.retry_max_delay is not None:
        updates["retry_max_delay"] = prefs.retry_max_delay
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
            emitter = _make_mcp_emitter(ctx) if ctx else None
            try:
                request = task.to_request()
                runner = RunnerFactory.create(task.cli)
                response = await runner.run(request, emitter=emitter)
                return AgentTaskResult(label=task.label, output=response.output)  # type: ignore[arg-type]
            except Exception as e:
                if emitter:
                    await emitter("error", f"Task '{task.label}' failed: {e}")
                else:
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
    retry_base_delay: float | None = None,
    retry_max_delay: float | None = None,
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
        retry_base_delay: Base delay seconds for exponential backoff (None inherits session/config).
        retry_max_delay: Backoff ceiling in seconds (None inherits session preference or config).
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
    # IMPORTANT: use `is not None`, NOT `or` — 0.0 is a valid value (instant backoff).
    resolved_retry_base_delay = (
        retry_base_delay if retry_base_delay is not None else prefs.retry_base_delay
    )
    resolved_retry_max_delay = (
        retry_max_delay if retry_max_delay is not None else prefs.retry_max_delay
    )

    task = AgentTask(
        cli=cli,
        prompt=prompt,
        context=context or {},
        execution_mode=resolved_mode,
        model=resolved_model,
        max_retries=resolved_max_retries,
        output_limit=resolved_output_limit,
        timeout=resolved_timeout,
        retry_base_delay=resolved_retry_base_delay,
        retry_max_delay=resolved_retry_max_delay,
    )
    result = await batch_prompt(tasks=[task], ctx=ctx)
    task_result = result.results[0]
    if task_result.error:
        raise ToolError(task_result.formatted_error)
    assert task_result.output is not None  # guaranteed: error is None, so output was set
    return task_result.output


async def set_preferences(
    execution_mode: ExecutionMode | None = None,
    model: str | None = None,
    max_retries: int | None = None,
    output_limit: int | None = None,
    timeout: int | None = None,
    retry_base_delay: float | None = None,
    retry_max_delay: float | None = None,
    clear_execution_mode: bool = False,
    clear_model: bool = False,
    clear_max_retries: bool = False,
    clear_output_limit: bool = False,
    clear_timeout: bool = False,
    clear_retry_base_delay: bool = False,
    clear_retry_max_delay: bool = False,
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
        retry_base_delay: Default base delay seconds for exponential backoff.
            None retains the current session value (use clear_retry_base_delay=True to reset).
        retry_max_delay: Default max delay cap seconds for exponential backoff.
            None retains the current session value (use clear_retry_max_delay=True to reset).
        clear_execution_mode: If True, resets execution_mode to None regardless of the
            execution_mode argument.
        clear_model: If True, resets model to None regardless of the model argument.
        clear_max_retries: If True, resets max_retries to None regardless of the argument.
        clear_output_limit: If True, resets output_limit to None regardless of the argument.
        clear_timeout: If True, resets timeout to None regardless of the argument.
        clear_retry_base_delay: If True, resets retry_base_delay to None.
        clear_retry_max_delay: If True, resets retry_max_delay to None.
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

    new_retry_base_delay: float | None
    if clear_retry_base_delay:
        new_retry_base_delay = None
    elif retry_base_delay is not None:
        new_retry_base_delay = retry_base_delay
    else:
        new_retry_base_delay = existing.retry_base_delay

    new_retry_max_delay: float | None
    if clear_retry_max_delay:
        new_retry_max_delay = None
    elif retry_max_delay is not None:
        new_retry_max_delay = retry_max_delay
    else:
        new_retry_max_delay = existing.retry_max_delay

    merged = SessionPreferences(
        execution_mode=new_execution_mode,
        model=new_model,
        max_retries=new_max_retries,
        output_limit=new_output_limit,
        timeout=new_timeout,
        retry_base_delay=new_retry_base_delay,
        retry_max_delay=new_retry_max_delay,
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


# Inject CLI names as enum into tool schemas before registration freezes them.
_inject_cli_enum()

# Register functions as MCP tools after definition so tests can import
# and call the raw functions directly (not the FunctionTool wrappers).
# timeout wraps synchronous calls with anyio.fail_after(); when a client
# passes task=True, the call goes through Docket instead (subprocess-level
# timeout applies). Both prompt and batch_prompt support either path.
_tool_timeout = get_tool_timeout()
mcp.tool(task=True, timeout=_tool_timeout)(batch_prompt)
mcp.tool(task=True, timeout=_tool_timeout)(prompt)
mcp.tool()(set_preferences)
mcp.tool()(get_preferences)
mcp.tool()(clear_preferences)
