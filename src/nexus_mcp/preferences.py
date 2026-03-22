"""Session preference management for nexus-mcp.

Provides three MCP tools (set/get/clear) and internal helpers for preference
resolution in prompt/batch_prompt.
"""

import json
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from nexus_mcp.types import (
    PREFERENCES_KEY,
    AgentTask,
    ExecutionMode,
    SessionPreferences,
)


async def _get_session_preferences(ctx: Context | None) -> SessionPreferences:
    """Read session preferences from ctx state, returning defaults when unset or ctx is None."""
    if ctx is None:
        return SessionPreferences()
    raw = await ctx.get_state(PREFERENCES_KEY)
    if raw is None:
        return SessionPreferences()
    try:
        return SessionPreferences(**raw)  # reconstruct from dict after JSON round-trip
    except (ValidationError, TypeError) as e:
        raise ToolError(f"Session preferences are corrupted and cannot be loaded: {e}") from e


_APPLY_FIELDS = (
    "execution_mode",
    "model",
    "max_retries",
    "output_limit",
    "timeout",
    "retry_base_delay",
    "retry_max_delay",
)


def _apply_preferences(task: AgentTask, prefs: SessionPreferences) -> AgentTask:
    """Fill in None fields on a task from session preferences.

    Returns the same task object if no updates are needed, otherwise a model_copy.
    Primary resolution happens in prompt(); this is the safety net for batch_prompt tasks.
    """
    updates: dict[str, Any] = {}
    for field in _APPLY_FIELDS:
        if getattr(task, field) is None:
            pref_val = getattr(prefs, field)
            if field == "execution_mode":
                updates[field] = pref_val or "default"
            elif pref_val is not None:
                updates[field] = pref_val
    return task.model_copy(update=updates) if updates else task


def _resolve_field(clear: bool, new_value: Any, existing_value: Any) -> Any:
    """Resolve a preference field: clear wins, then new value, then existing."""
    if clear:
        return None
    return new_value if new_value is not None else existing_value


async def set_preferences(
    execution_mode: ExecutionMode | None = None,
    model: str | None = None,
    max_retries: int | None = None,
    output_limit: int | None = None,
    timeout: int | None = None,
    retry_base_delay: float | None = None,
    retry_max_delay: float | None = None,
    elicit: bool | None = None,
    confirm_yolo: bool | None = None,
    confirm_vague_prompt: bool | None = None,
    confirm_high_retries: bool | None = None,
    confirm_large_batch: bool | None = None,
    clear_execution_mode: bool = False,
    clear_model: bool = False,
    clear_max_retries: bool = False,
    clear_output_limit: bool = False,
    clear_timeout: bool = False,
    clear_retry_base_delay: bool = False,
    clear_retry_max_delay: bool = False,
    clear_elicit: bool = False,
    clear_confirm_yolo: bool = False,
    clear_confirm_vague_prompt: bool = False,
    clear_confirm_high_retries: bool = False,
    clear_confirm_large_batch: bool = False,
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

    merged = SessionPreferences(
        execution_mode=_resolve_field(
            clear_execution_mode, execution_mode, existing.execution_mode
        ),
        model=_resolve_field(clear_model, model, existing.model),
        max_retries=_resolve_field(clear_max_retries, max_retries, existing.max_retries),
        output_limit=_resolve_field(clear_output_limit, output_limit, existing.output_limit),
        timeout=_resolve_field(clear_timeout, timeout, existing.timeout),
        retry_base_delay=_resolve_field(
            clear_retry_base_delay, retry_base_delay, existing.retry_base_delay
        ),
        retry_max_delay=_resolve_field(
            clear_retry_max_delay, retry_max_delay, existing.retry_max_delay
        ),
        elicit=_resolve_field(clear_elicit, elicit, existing.elicit),
        confirm_yolo=_resolve_field(clear_confirm_yolo, confirm_yolo, existing.confirm_yolo),
        confirm_vague_prompt=_resolve_field(
            clear_confirm_vague_prompt, confirm_vague_prompt, existing.confirm_vague_prompt
        ),
        confirm_high_retries=_resolve_field(
            clear_confirm_high_retries, confirm_high_retries, existing.confirm_high_retries
        ),
        confirm_large_batch=_resolve_field(
            clear_confirm_large_batch, confirm_large_batch, existing.confirm_large_batch
        ),
    )
    await ctx.set_state(PREFERENCES_KEY, merged.model_dump())
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
    await ctx.delete_state(PREFERENCES_KEY)
    return "Preferences cleared"
