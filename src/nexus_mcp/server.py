# src/nexus_mcp/server.py
"""FastMCP server with CLI agent tools.

Exposes four MCP tools:
- batch_prompt: Send multiple prompts to CLI agents in parallel (primary tool)
- prompt: Send a single prompt to a CLI agent, routes to batch_prompt
- set_preferences: Set session defaults (execution mode, model, retries, etc.)
- clear_preferences: Reset all session preferences

Session preferences are readable via the nexus://preferences MCP resource.

Background task design: both prompt and batch_prompt use @mcp.tool(task=True) so they
run asynchronously and return task IDs immediately. This prevents MCP timeouts for long
operations like YOLO mode (2-5 minutes).

Testability: The tool functions are defined as plain async functions and then
registered with @mcp.tool so tests can import and call them directly without
going through the FunctionTool wrapper.
"""

import asyncio
import contextlib
import json as _json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from nexus_mcp.cli_detector import detect_cli
from nexus_mcp.compound_tools import register_compound_tools
from nexus_mcp.config import get_runner_defaults, get_runner_models, get_tool_timeout
from nexus_mcp.elicitation import ElicitationGuard
from nexus_mcp.emitters import make_mcp_emitter, make_progress_emitter
from nexus_mcp.http_client import get_http_client
from nexus_mcp.icons import SERVER_ICONS, TOOL_CONFIG_ICONS, TOOL_EXEC_ICONS
from nexus_mcp.labels import assign_labels
from nexus_mcp.middleware import (
    ErrorNormalizationMiddleware,
    RequestLoggingMiddleware,
    TimingMiddleware,
)
from nexus_mcp.openapi_setup import setup_opencode_tools
from nexus_mcp.opencode_resources import (
    is_opencode_server_configured,
    register_opencode_data_resources,
    register_opencode_status_resource,
)
from nexus_mcp.preferences import (
    _apply_preferences,
    _get_session_preferences,
    clear_preferences,
    set_preferences,
)
from nexus_mcp.prompts import register_prompts
from nexus_mcp.resources import register_resources
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.store import save_model_tiers
from nexus_mcp.types import (
    DEFAULT_MAX_CONCURRENCY,
    AgentTask,
    AgentTaskResult,
    ExecutionMode,
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
    lines.append("## Important: Do not pre-fill `cli` or `model`")
    lines.append("Leave `cli` and `model` empty so the user can choose interactively.")
    lines.append("Only set them when the user explicitly names a runner or model.")
    lines.append("")
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

    lines.append("## Model Benchmark Data Sources")
    lines.append("")
    lines.append("Use these to inform runner and model selection (no API keys required):")
    lines.append("")
    lines.append("- Artificial Analysis: https://artificialanalysis.ai/leaderboards/models")
    lines.append("- OpenRouter: https://openrouter.ai/api/v1/models")
    lines.append("- Chatbot Arena: https://lmarena.ai/?leaderboard")
    lines.append("- LLM Stats: https://llm-stats.com")
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
    prompt.__annotations__["cli"] = Annotated[
        str | None, Field(default=None, json_schema_extra={"enum": cli_names})
    ]

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


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Conditionally register OpenCode tools based on server availability."""
    register_opencode_status_resource(server)

    client_to_close = None
    if is_opencode_server_configured():
        client = get_http_client()
        client_to_close = client

        server.tool(annotations=_CONFIG_OC_ANNOTATIONS, tags={"configuration"})(
            opencode_set_provider_auth
        )
        server.tool(annotations=_CONFIG_OC_ANNOTATIONS, tags={"configuration"})(
            opencode_update_config
        )

        healthy = await client.health_check()
        if healthy:
            await setup_opencode_tools(server, client)
            register_compound_tools(server)
            register_opencode_data_resources(server)
            logger.info("OpenCode server tools registered (server healthy)")
        else:
            logger.warning("OpenCode server not reachable, mutation tools only")
    else:
        logger.warning("OpenCode server not configured (NEXUS_OPENCODE_SERVER_PASSWORD not set)")

    yield

    if client_to_close is not None:
        with contextlib.suppress(Exception):
            await client_to_close.close()


mcp = FastMCP(
    "nexus-mcp",
    instructions=build_server_instructions(),
    icons=SERVER_ICONS,
    lifespan=_lifespan,
)

# Middleware executes outermost → innermost on request, reverse on response.
# Order: ErrorNormalization (catch all) → Timing (measure) → RequestLogging (log entry/exit)
mcp.add_middleware(ErrorNormalizationMiddleware())
mcp.add_middleware(TimingMiddleware())
mcp.add_middleware(RequestLoggingMiddleware())

logger = logging.getLogger(__name__)


def _resolve_elicit(elicit: bool | None, prefs: SessionPreferences) -> bool:
    """Resolve the effective elicit flag from explicit arg and session preferences.

    Priority: explicit argument > session preference > default (True).
    """
    if elicit is not None:
        return elicit
    if prefs.elicit is not None:
        return prefs.elicit
    return True


async def batch_prompt(
    *,
    tasks: list[AgentTask],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    elicit: bool | None = None,
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

    resolved_elicit = _resolve_elicit(elicit, prefs)
    guard = (
        ElicitationGuard(ctx, installed_clis=list(RunnerFactory.list_clis()), prefs=prefs)
        if ctx
        else None
    )
    if guard:
        tasks = await guard.check_batch(tasks, elicit=resolved_elicit)
    else:
        for t in tasks:
            if t.cli is None:
                raise ToolError("cli is required on all tasks when no context available")

    def _metadata_header(task: AgentTask) -> str:
        """Build a short metadata line so the AI knows which runner handled the task."""
        model_part = task.model or "default"
        return f"[cli: {task.cli} | model: {model_part} | mode: {task.execution_mode}]"

    labelled = assign_labels(tasks)
    semaphore = asyncio.Semaphore(max_concurrency)
    is_single_task = len(labelled) == 1

    if ctx:
        await ctx.info(f"Starting batch of {len(labelled)} tasks (concurrency={max_concurrency})")

    async def _run_single(idx: int, task: AgentTask) -> AgentTaskResult:
        async with semaphore:
            emitter = make_mcp_emitter(ctx) if ctx else None
            progress = None
            if ctx:
                if is_single_task:
                    progress = make_progress_emitter(ctx)
                else:
                    progress = make_progress_emitter(
                        ctx,
                        task_idx=idx + 1,
                        task_count=len(labelled),
                        label=task.label,
                    )
            try:
                request = task.to_request()
                runner = RunnerFactory.create(request.cli)
                response = await runner.run(request, emitter=emitter, progress=progress)
                header = _metadata_header(task)
                output = f"{header}\n\n{response.output}"
                return AgentTaskResult(label=task.label, output=output)  # type: ignore[arg-type]
            except Exception as e:
                if emitter:
                    await emitter("error", f"Task '{task.label}' failed: {e}")
                else:
                    logger.exception("Task %r failed: %s", task.label, e)
                return AgentTaskResult(label=task.label, error=str(e), error_type=type(e).__name__)  # type: ignore[arg-type]

    results = await asyncio.gather(*[_run_single(i, t) for i, t in enumerate(labelled)])
    response = MultiPromptResponse(results=list(results))
    if ctx:
        await ctx.info(f"Batch complete: {response.succeeded}/{response.total} succeeded")
    return response


async def prompt(
    *,
    cli: str | None = None,
    prompt: str,
    context: dict[str, Any] | None = None,
    execution_mode: ExecutionMode | None = None,
    model: str | None = None,
    max_retries: int | None = None,
    output_limit: int | None = None,
    timeout: int | None = None,
    retry_base_delay: float | None = None,
    retry_max_delay: float | None = None,
    elicit: bool | None = None,
    ctx: Context | None = None,
) -> str:
    """Send a prompt to a CLI runner as a background task.

    Returns immediately with a task ID. Client polls for results.
    This prevents timeouts for long operations (YOLO mode: 2-5 minutes).

    Args:
        cli: CLI runner name (e.g., "gemini"). None triggers interactive selection.
        prompt: Prompt text to send to the runner
        context: Optional context metadata
        execution_mode: 'default' (safe) or 'yolo'. None inherits session preference.
        model: Model name. None triggers interactive selection or uses CLI default.
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

    resolved_elicit = _resolve_elicit(elicit, prefs)
    guard = (
        ElicitationGuard(ctx, installed_clis=list(RunnerFactory.list_clis()), prefs=prefs)
        if ctx
        else None
    )
    selections: dict[str, str] = {}
    if guard:
        resolved = await guard.check_prompt(
            cli=cli,
            model=resolved_model,
            execution_mode=resolved_mode,
            prompt_text=prompt,
            elicit=resolved_elicit,
        )
        cli = resolved.cli
        resolved_model = resolved.model
        resolved_mode = resolved.execution_mode
        prompt = resolved.prompt_text
        selections = resolved.selections
    elif cli is None:
        raise ToolError("cli is required")

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
    result = await batch_prompt(tasks=[task], elicit=False, ctx=ctx)
    task_result = result.results[0]
    if task_result.error:
        raise ToolError(task_result.formatted_error)
    assert task_result.output is not None  # guaranteed: error is None, so output was set

    # Annotate per-field using elicitation selections.
    output = task_result.output
    field_map = {"cli": cli, "model": resolved_model or "default", "mode": resolved_mode}
    for field, value in field_map.items():
        status = selections.get(field)
        if status:
            output = output.replace(f"{field}: {value}", f"{field}: {value} ({status})", 1)
    return output


async def set_model_tiers(
    *,
    tiers: dict[str, str],
    ctx: Context | None = None,
) -> str:
    """Save model tier classifications.

    Client sends sampling/benchmark results; server persists to backing store.
    Overwrites any previously saved tiers entirely.

    Args:
        tiers: Mapping of model name to tier ('quick', 'standard', 'thorough').
        ctx: MCP context (auto-injected by FastMCP).

    Returns:
        Confirmation string with the number of tiers saved.
    """
    if ctx is None:
        raise ToolError("set_model_tiers requires an active session context")
    await save_model_tiers(ctx, tiers)
    return f"Model tiers saved: {len(tiers)} model(s) classified"


async def opencode_set_provider_auth(
    *,
    provider_id: str,
    credentials: dict[str, Any],
) -> str:
    """Set authentication credentials for a provider."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", provider_id):
        raise ToolError(f"Invalid provider_id: {provider_id!r}")
    await get_http_client().put(f"/auth/{provider_id}", json=credentials)
    return f"Credentials set for provider '{provider_id}'"


async def opencode_update_config(
    *,
    config: dict[str, Any],
) -> str:
    """Update OpenCode server configuration."""
    data = await get_http_client().patch("/config", json=config)
    return _json.dumps(data, indent=2)


# Inject CLI names as enum into tool schemas before registration freezes them.
_inject_cli_enum()

# Register functions as MCP tools after definition so tests can import
# and call the raw functions directly (not the FunctionTool wrappers).
# timeout wraps synchronous calls with anyio.fail_after(); when a client
# passes task=True, the call goes through Docket instead (subprocess-level
# timeout applies). Both prompt and batch_prompt support either path.
_tool_timeout = get_tool_timeout()

# Annotations communicate behavioral hints to MCP clients (e.g. auto-approval decisions).
_EXEC_ANNOTATIONS = ToolAnnotations(
    title="Prompt CLI Agent",
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=True,
)
_BATCH_EXEC_ANNOTATIONS = ToolAnnotations(
    title="Batch Prompt CLI Agents",
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=True,
)
_SET_PREFS_ANNOTATIONS = ToolAnnotations(
    title="Set Session Preferences",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
_CLEAR_PREFS_ANNOTATIONS = ToolAnnotations(
    title="Clear Session Preferences",
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=True,
    openWorldHint=False,
)
_SET_TIERS_ANNOTATIONS = ToolAnnotations(
    title="Set Model Tiers",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
mcp.tool(
    task=True,
    timeout=_tool_timeout,
    icons=TOOL_EXEC_ICONS,
    annotations=_BATCH_EXEC_ANNOTATIONS,
    tags={"agent-execution"},
)(batch_prompt)
mcp.tool(
    task=True,
    timeout=_tool_timeout,
    icons=TOOL_EXEC_ICONS,
    annotations=_EXEC_ANNOTATIONS,
    tags={"agent-execution"},
)(prompt)
mcp.tool(
    icons=TOOL_CONFIG_ICONS,
    annotations=_SET_PREFS_ANNOTATIONS,
    tags={"configuration"},
)(set_preferences)
mcp.tool(
    icons=TOOL_CONFIG_ICONS,
    annotations=_CLEAR_PREFS_ANNOTATIONS,
    tags={"configuration"},
)(clear_preferences)
mcp.tool(
    icons=TOOL_CONFIG_ICONS,
    annotations=_SET_TIERS_ANNOTATIONS,
    tags={"configuration"},
)(set_model_tiers)
_CONFIG_OC_ANNOTATIONS = ToolAnnotations(
    title="OpenCode Configuration",
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)

# Register MCP resources (read-only data endpoints).
register_resources(mcp)

# Register MCP prompt templates (discoverable workflow scaffolds).
register_prompts(mcp)
