# tests/unit/test_preferences.py
"""Unit tests for session preference tools and preference fallback in prompt/batch_prompt."""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.server import (
    _apply_preferences,
    _get_session_preferences,
    batch_prompt,
    clear_preferences,
    get_preferences,
    prompt,
    set_preferences,
)
from nexus_mcp.types import AgentTask, SessionPreferences
from tests.fixtures import make_agent_response, make_session_preferences

_PREFS_KEY = "nexus:preferences"


def _setup_mock_runner(mock_factory, *, output: str = "test output", side_effect=None) -> AsyncMock:
    mock_runner = AsyncMock()
    if side_effect is not None:
        mock_runner.run.side_effect = side_effect
    else:
        mock_runner.run.return_value = make_agent_response(output=output)
    mock_factory.create.return_value = mock_runner
    return mock_runner


# ---------------------------------------------------------------------------
# TestGetSessionPreferences helper
# ---------------------------------------------------------------------------


class TestGetSessionPreferences:
    async def test_returns_empty_prefs_when_ctx_none(self):
        prefs = await _get_session_preferences(None)
        assert prefs.execution_mode is None
        assert prefs.model is None

    async def test_returns_empty_prefs_when_no_state(self, ctx):
        ctx.get_state.return_value = None
        prefs = await _get_session_preferences(ctx)
        assert prefs.execution_mode is None
        assert prefs.model is None

    async def test_reconstructs_from_dict(self, ctx):
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
        prefs = await _get_session_preferences(ctx)
        assert prefs.execution_mode == "yolo"
        assert prefs.model == "gemini-2.5-flash"

    async def test_raises_tool_error_for_corrupted_state(self, ctx):
        """Malformed session state raises ToolError instead of crashing with ValidationError."""
        ctx.get_state.return_value = {"execution_mode": "invalid_value"}
        with pytest.raises(ToolError, match="corrupted"):
            await _get_session_preferences(ctx)


# ---------------------------------------------------------------------------
# TestApplyPreferences helper
# ---------------------------------------------------------------------------


class TestApplyPreferences:
    def test_fills_none_execution_mode_from_prefs(self):
        task = AgentTask(agent="gemini", prompt="hi", execution_mode=None)
        prefs = make_session_preferences(execution_mode="yolo")
        result = _apply_preferences(task, prefs)
        assert result.execution_mode == "yolo"

    def test_fills_none_execution_mode_with_default_when_prefs_none(self):
        task = AgentTask(agent="gemini", prompt="hi", execution_mode=None)
        prefs = make_session_preferences()  # both None
        result = _apply_preferences(task, prefs)
        assert result.execution_mode == "default"

    def test_does_not_override_explicit_execution_mode(self):
        task = AgentTask(agent="gemini", prompt="hi", execution_mode="sandbox")
        prefs = make_session_preferences(execution_mode="yolo")
        result = _apply_preferences(task, prefs)
        assert result.execution_mode == "sandbox"

    def test_fills_none_model_from_prefs(self):
        task = AgentTask(agent="gemini", prompt="hi", model=None)
        prefs = make_session_preferences(model="gemini-2.5-flash")
        result = _apply_preferences(task, prefs)
        assert result.model == "gemini-2.5-flash"

    def test_does_not_override_explicit_model(self):
        task = AgentTask(agent="gemini", prompt="hi", model="gemini-1.5-pro")
        prefs = make_session_preferences(model="gemini-2.5-flash")
        result = _apply_preferences(task, prefs)
        assert result.model == "gemini-1.5-pro"

    def test_does_not_fill_model_when_prefs_model_none(self):
        task = AgentTask(agent="gemini", prompt="hi", model=None)
        prefs = make_session_preferences()  # model=None
        result = _apply_preferences(task, prefs)
        assert result.model is None

    def test_returns_equivalent_task_when_no_updates(self):
        task = AgentTask(agent="gemini", prompt="hi", execution_mode="default")
        prefs = make_session_preferences()
        result = _apply_preferences(task, prefs)
        assert result == task


# ---------------------------------------------------------------------------
# TestSetPreferences
# ---------------------------------------------------------------------------


class TestSetPreferences:
    async def test_raises_tool_error_when_ctx_none(self):
        with pytest.raises(ToolError):
            await set_preferences(execution_mode="yolo", ctx=None)

    async def test_sets_execution_mode(self, ctx):
        ctx.get_state.return_value = None  # no existing prefs
        result = await set_preferences(execution_mode="yolo", ctx=ctx)
        ctx.set_state.assert_awaited_once_with(
            _PREFS_KEY, {"execution_mode": "yolo", "model": None}
        )
        assert "yolo" in result

    async def test_sets_model(self, ctx):
        ctx.get_state.return_value = None
        await set_preferences(model="gemini-2.5-flash", ctx=ctx)
        ctx.set_state.assert_awaited_once_with(
            _PREFS_KEY, {"execution_mode": None, "model": "gemini-2.5-flash"}
        )

    async def test_sets_both(self, ctx):
        ctx.get_state.return_value = None
        await set_preferences(execution_mode="sandbox", model="gemini-2.5-flash", ctx=ctx)
        ctx.set_state.assert_awaited_once_with(
            _PREFS_KEY, {"execution_mode": "sandbox", "model": "gemini-2.5-flash"}
        )

    async def test_merges_with_existing_prefs(self, ctx):
        """Setting only model preserves existing execution_mode."""
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": None}
        await set_preferences(model="gemini-2.5-flash", ctx=ctx)
        ctx.set_state.assert_awaited_once_with(
            _PREFS_KEY, {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
        )

    async def test_all_none_params_preserves_existing(self, ctx):
        """set_preferences() with no args keeps existing state intact."""
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
        await set_preferences(ctx=ctx)
        ctx.set_state.assert_awaited_once_with(
            _PREFS_KEY, {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
        )

    async def test_returns_json_formatted_confirmation(self, ctx):
        """Return string contains JSON (null not None, double-quoted keys)."""
        import json

        ctx.get_state.return_value = None
        result = await set_preferences(execution_mode="default", ctx=ctx)
        assert isinstance(result, str)
        # Extract the JSON portion after "Preferences set: "
        json_part = result.split("Preferences set: ", 1)[1]
        parsed = json.loads(json_part)  # raises if not valid JSON
        assert parsed["execution_mode"] == "default"

    async def test_clear_execution_mode_resets_to_none(self, ctx):
        """clear_execution_mode=True resets execution_mode while keeping model."""
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
        await set_preferences(clear_execution_mode=True, ctx=ctx)
        ctx.set_state.assert_awaited_once_with(
            _PREFS_KEY, {"execution_mode": None, "model": "gemini-2.5-flash"}
        )

    async def test_clear_model_resets_to_none(self, ctx):
        """clear_model=True resets model while keeping execution_mode."""
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
        await set_preferences(clear_model=True, ctx=ctx)
        ctx.set_state.assert_awaited_once_with(
            _PREFS_KEY, {"execution_mode": "yolo", "model": None}
        )

    async def test_clear_flag_takes_precedence_over_value(self, ctx):
        """clear_execution_mode=True ignores any passed execution_mode value."""
        ctx.get_state.return_value = None
        await set_preferences(execution_mode="yolo", clear_execution_mode=True, ctx=ctx)
        ctx.set_state.assert_awaited_once_with(_PREFS_KEY, {"execution_mode": None, "model": None})

    async def test_returns_confirmation_string(self, ctx):
        ctx.get_state.return_value = None
        result = await set_preferences(execution_mode="default", ctx=ctx)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestGetPreferences
# ---------------------------------------------------------------------------


class TestGetPreferences:
    async def test_raises_tool_error_when_ctx_none(self):
        with pytest.raises(ToolError):
            await get_preferences(ctx=None)

    async def test_returns_prefs_dict(self, ctx):
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
        result = await get_preferences(ctx=ctx)
        assert result == {"execution_mode": "yolo", "model": "gemini-2.5-flash"}

    async def test_returns_empty_defaults_when_no_prefs(self, ctx):
        ctx.get_state.return_value = None
        result = await get_preferences(ctx=ctx)
        assert result == {"execution_mode": None, "model": None}

    async def test_returns_dict_not_pydantic(self, ctx):
        ctx.get_state.return_value = None
        result = await get_preferences(ctx=ctx)
        assert isinstance(result, dict)
        assert not isinstance(result, SessionPreferences)


# ---------------------------------------------------------------------------
# TestClearPreferences
# ---------------------------------------------------------------------------


class TestClearPreferences:
    async def test_raises_tool_error_when_ctx_none(self):
        with pytest.raises(ToolError):
            await clear_preferences(ctx=None)

    async def test_calls_delete_state(self, ctx):
        result = await clear_preferences(ctx=ctx)
        ctx.delete_state.assert_awaited_once_with(_PREFS_KEY)
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_returns_confirmation_string(self, ctx):
        result = await clear_preferences(ctx=ctx)
        assert "cleared" in result.lower()


# ---------------------------------------------------------------------------
# TestPromptPreferenceFallback
# ---------------------------------------------------------------------------


class TestPromptPreferenceFallback:
    @patch("nexus_mcp.server.RunnerFactory")
    async def test_session_mode_used_when_no_explicit_mode(self, mock_factory, ctx):
        """Session execution_mode='yolo' is used when prompt() called without explicit mode."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": None}

        await prompt(agent="gemini", prompt="test", ctx=ctx)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "yolo"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_explicit_mode_overrides_session(self, mock_factory, ctx):
        """Explicit execution_mode='sandbox' overrides session 'yolo'."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": None}

        await prompt(agent="gemini", prompt="test", execution_mode="sandbox", ctx=ctx)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "sandbox"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_session_model_used_when_no_explicit_model(self, mock_factory, ctx):
        """Session model is used when prompt() called without explicit model."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")
        ctx.get_state.return_value = {"execution_mode": None, "model": "gemini-2.5-flash"}

        await prompt(agent="gemini", prompt="test", ctx=ctx)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.model == "gemini-2.5-flash"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_explicit_model_overrides_session(self, mock_factory, ctx):
        """Explicit model overrides session model."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")
        ctx.get_state.return_value = {"execution_mode": None, "model": "gemini-2.5-flash"}

        await prompt(agent="gemini", prompt="test", model="gemini-1.5-pro", ctx=ctx)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.model == "gemini-1.5-pro"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_no_session_no_explicit_uses_defaults(self, mock_factory):
        """Without session or explicit params, execution_mode='default' and model=None."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")

        await prompt(agent="gemini", prompt="test")  # ctx=None → no session

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "default"
        assert call_args.model is None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_ctx_none_does_not_crash(self, mock_factory):
        """prompt() with ctx=None falls back to defaults without raising."""
        _setup_mock_runner(mock_factory, output="ok")
        result = await prompt(agent="gemini", prompt="test", ctx=None)
        assert result == "ok"


# ---------------------------------------------------------------------------
# TestBatchPromptPreferenceFallback
# ---------------------------------------------------------------------------


class TestBatchPromptPreferenceFallback:
    @patch("nexus_mcp.server.RunnerFactory")
    async def test_session_mode_applied_to_tasks_with_none_mode(self, mock_factory, ctx):
        """Session execution_mode='yolo' is applied to tasks with execution_mode=None."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": None}

        tasks = [AgentTask(agent="gemini", prompt="test", execution_mode=None)]
        await batch_prompt(tasks=tasks, ctx=ctx)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "yolo"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_explicit_task_mode_not_overridden(self, mock_factory, ctx):
        """Task with explicit execution_mode='sandbox' keeps it despite session 'yolo'."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": None}

        tasks = [AgentTask(agent="gemini", prompt="test", execution_mode="sandbox")]
        await batch_prompt(tasks=tasks, ctx=ctx)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "sandbox"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_session_model_applied_to_tasks(self, mock_factory, ctx):
        """Session model is applied to tasks with model=None."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")
        ctx.get_state.return_value = {"execution_mode": None, "model": "gemini-2.5-flash"}

        tasks = [AgentTask(agent="gemini", prompt="test")]
        await batch_prompt(tasks=tasks, ctx=ctx)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.model == "gemini-2.5-flash"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_no_session_resolves_none_to_default(self, mock_factory):
        """Without session, task with execution_mode=None resolves to 'default'."""
        mock_runner = _setup_mock_runner(mock_factory, output="ok")

        tasks = [AgentTask(agent="gemini", prompt="test", execution_mode=None)]
        await batch_prompt(tasks=tasks)  # ctx=None

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "default"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_mixed_tasks_selective_apply(self, mock_factory, ctx):
        """Session mode applies to None-mode tasks; explicit-mode tasks keep theirs."""
        call_modes: list[str] = []

        async def capture_mode(request):
            call_modes.append(request.execution_mode)
            return make_agent_response()

        mock_factory.create.return_value = AsyncMock()
        mock_factory.create.return_value.run.side_effect = capture_mode
        ctx.get_state.return_value = {"execution_mode": "yolo", "model": None}

        tasks = [
            AgentTask(agent="gemini", prompt="t1", execution_mode=None),
            AgentTask(agent="gemini", prompt="t2", execution_mode="sandbox"),
        ]
        await batch_prompt(tasks=tasks, ctx=ctx)

        assert "yolo" in call_modes
        assert "sandbox" in call_modes
