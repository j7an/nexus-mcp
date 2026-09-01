# tests/unit/test_preferences.py
"""Unit tests for persistent preference tools and preference fallback in prompt/batch_prompt."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.core import (
    ExecutionConfigValues,
    JobEvent,
    JobResultEnvelope,
    SucceededJobResultResponse,
    TurnOperation,
    TurnResult,
)
from nexus_mcp.jobs import AgentJobService
from nexus_mcp.preferences import (
    _apply_preferences,
    _get_session_preferences,
    clear_preferences,
    get_preferences,
    set_preferences,
)
from nexus_mcp.server import batch_prompt, prompt
from nexus_mcp.types import AgentTask, SessionPreferences
from tests.fixtures import (
    REPRESENTATIVE_CLI,
    make_job_handle,
    make_session_preferences,
    strip_runner_header,
)

_LOAD = "nexus_mcp.mcp.preferences.load_preferences"
_SAVE = "nexus_mcp.mcp.preferences.save_preferences"
_DELETE = "nexus_mcp.mcp.preferences._delete_prefs"

_ALL_NONE_PREFS = {
    "execution_mode": None,
    "model": None,
    "max_retries": None,
    "output_limit": None,
    "timeout": None,
    "retry_base_delay": None,
    "retry_max_delay": None,
    "elicit": None,
    "confirm_yolo": None,
    "confirm_vague_prompt": None,
    "confirm_high_retries": None,
    "confirm_large_batch": None,
}


class _PreferenceJobServiceBoundary:
    """Durable service boundary for compatibility preference assertions."""

    def __init__(self) -> None:
        self.operations: dict[str, TurnOperation] = {}
        self.start = AsyncMock(side_effect=self._start)
        self.result = AsyncMock(side_effect=self._result)
        self.subscribe_events = Mock(side_effect=self._subscribe_events)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(AgentJobService, "start", self.start)
        monkeypatch.setattr(AgentJobService, "result", self.result)
        monkeypatch.setattr(AgentJobService, "subscribe_events", self.subscribe_events)

    async def _start(self, **kwargs):
        operation = kwargs["operation"]
        job_id = f"job-{len(self.operations) + 1}"
        self.operations[job_id] = operation
        return make_job_handle(
            job_id=job_id,
            session_id=f"session-{len(self.operations)}",
            operation=operation,
        )

    def _subscribe_events(self, **kwargs):
        job_id = kwargs["job_id"]

        async def events():
            yield JobEvent(job_id=job_id, sequence=1, type="job_completed")

        return events()

    async def _result(self, **kwargs):
        job_id = kwargs["job_id"]
        return SucceededJobResultResponse(
            job_id=job_id,
            result=JobResultEnvelope(
                job_id=job_id,
                payload=TurnResult(message="ok"),
            ),
        )

    def config_for_prompt(self, prompt_text: str) -> ExecutionConfigValues:
        matches = [
            call
            for call in self.start.await_args_list
            if call.kwargs["operation"].prompt == prompt_text
        ]
        assert len(matches) == 1
        assert matches[0].kwargs["operation"] == TurnOperation(prompt=prompt_text)
        return matches[0].kwargs["explicit_config"]

    def only_config_for_prompt(self, prompt_text: str) -> ExecutionConfigValues:
        assert self.prompt_order() == [prompt_text]
        return self.config_for_prompt(prompt_text)

    def prompt_order(self) -> list[str]:
        return [call.kwargs["operation"].prompt for call in self.start.await_args_list]


@pytest.fixture
def preference_job_service_boundary(
    monkeypatch, fake_runner_registry
) -> _PreferenceJobServiceBoundary:
    del fake_runner_registry
    boundary = _PreferenceJobServiceBoundary()
    boundary.install(monkeypatch)
    return boundary


# ---------------------------------------------------------------------------
# TestGetSessionPreferences helper
# ---------------------------------------------------------------------------


class TestGetSessionPreferences:
    async def test_returns_empty_prefs_when_ctx_none(self):
        prefs = await _get_session_preferences(None)
        assert prefs.execution_mode is None
        assert prefs.model is None

    @patch(_LOAD)
    async def test_returns_empty_prefs_when_no_state(self, mock_load, ctx):
        mock_load.return_value = None
        prefs = await _get_session_preferences(ctx)
        assert prefs.execution_mode is None
        assert prefs.model is None

    @patch(_LOAD)
    async def test_reconstructs_from_dict(self, mock_load, ctx):
        mock_load.return_value = {"execution_mode": "yolo", "model": "test-model"}
        prefs = await _get_session_preferences(ctx)
        assert prefs.execution_mode == "yolo"
        assert prefs.model == "test-model"

    @patch(_LOAD)
    async def test_raises_tool_error_for_corrupted_state(self, mock_load, ctx):
        """Malformed stored state raises ToolError instead of crashing with ValidationError."""
        mock_load.return_value = {"execution_mode": "invalid_value"}
        with pytest.raises(ToolError, match="corrupted"):
            await _get_session_preferences(ctx)


# ---------------------------------------------------------------------------
# TestApplyPreferences helper
# ---------------------------------------------------------------------------


class TestApplyPreferences:
    def test_fills_none_execution_mode_from_prefs(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", execution_mode=None)
        prefs = make_session_preferences(execution_mode="yolo")
        result = _apply_preferences(task, prefs)
        assert result.execution_mode == "yolo"

    def test_fills_none_execution_mode_with_default_when_prefs_none(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", execution_mode=None)
        prefs = make_session_preferences()  # both None
        result = _apply_preferences(task, prefs)
        assert result.execution_mode == "default"

    def test_does_not_override_explicit_execution_mode(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", execution_mode="yolo")
        prefs = make_session_preferences(execution_mode="default")
        result = _apply_preferences(task, prefs)
        assert result.execution_mode == "yolo"

    def test_fills_none_model_from_prefs(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", model=None)
        prefs = make_session_preferences(model="test-model")
        result = _apply_preferences(task, prefs)
        assert result.model == "test-model"

    def test_does_not_override_explicit_model(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", model="alternate-model")
        prefs = make_session_preferences(model="test-model")
        result = _apply_preferences(task, prefs)
        assert result.model == "alternate-model"

    def test_does_not_fill_model_when_prefs_model_none(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", model=None)
        prefs = make_session_preferences()  # model=None
        result = _apply_preferences(task, prefs)
        assert result.model is None

    def test_returns_equivalent_task_when_no_updates(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", execution_mode="default")
        prefs = make_session_preferences()
        result = _apply_preferences(task, prefs)
        assert result == task

    def test_fills_none_max_retries_from_prefs(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi")
        prefs = make_session_preferences(max_retries=5)
        result = _apply_preferences(task, prefs)
        assert result.max_retries == 5

    def test_does_not_override_explicit_max_retries(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", max_retries=2)
        prefs = make_session_preferences(max_retries=5)
        result = _apply_preferences(task, prefs)
        assert result.max_retries == 2

    def test_fills_none_output_limit_from_prefs(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi")
        prefs = make_session_preferences(output_limit=4096)
        result = _apply_preferences(task, prefs)
        assert result.output_limit == 4096

    def test_does_not_override_explicit_output_limit(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", output_limit=1024)
        prefs = make_session_preferences(output_limit=4096)
        result = _apply_preferences(task, prefs)
        assert result.output_limit == 1024

    def test_fills_none_timeout_from_prefs(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi")
        prefs = make_session_preferences(timeout=30)
        result = _apply_preferences(task, prefs)
        assert result.timeout == 30

    def test_does_not_override_explicit_timeout(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", timeout=10)
        prefs = make_session_preferences(timeout=30)
        result = _apply_preferences(task, prefs)
        assert result.timeout == 10

    def test_fills_none_retry_base_delay_from_prefs(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi")
        prefs = make_session_preferences(retry_base_delay=1.5)
        result = _apply_preferences(task, prefs)
        assert result.retry_base_delay == 1.5

    def test_does_not_override_explicit_retry_base_delay(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", retry_base_delay=0.5)
        prefs = make_session_preferences(retry_base_delay=1.5)
        result = _apply_preferences(task, prefs)
        assert result.retry_base_delay == 0.5

    def test_zero_retry_base_delay_not_overridden(self):
        """0.0 is a valid delay — must not be treated as falsy and replaced by prefs."""
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi", retry_base_delay=0.0)
        prefs = make_session_preferences(retry_base_delay=2.0)
        result = _apply_preferences(task, prefs)
        assert result.retry_base_delay == 0.0

    def test_fills_none_retry_max_delay_from_prefs(self):
        task = AgentTask(cli=REPRESENTATIVE_CLI, prompt="hi")
        prefs = make_session_preferences(retry_max_delay=60.0)
        result = _apply_preferences(task, prefs)
        assert result.retry_max_delay == 60.0


# ---------------------------------------------------------------------------
# TestSetPreferences
# ---------------------------------------------------------------------------


class TestSetPreferences:
    async def test_raises_tool_error_when_ctx_none(self):
        with pytest.raises(ToolError):
            await set_preferences(execution_mode="yolo", ctx=None)

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_sets_execution_mode(self, mock_load, mock_save, ctx):
        mock_load.return_value = None  # no existing prefs
        result = await set_preferences(execution_mode="yolo", ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "execution_mode": "yolo"})
        assert "yolo" in result

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_sets_model(self, mock_load, mock_save, ctx):
        mock_load.return_value = None
        await set_preferences(model="test-model", ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "model": "test-model"})

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_sets_both(self, mock_load, mock_save, ctx):
        mock_load.return_value = None
        await set_preferences(execution_mode="yolo", model="test-model", ctx=ctx)
        mock_save.assert_awaited_once_with(
            ctx,
            {**_ALL_NONE_PREFS, "execution_mode": "yolo", "model": "test-model"},
        )

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_merges_with_existing_prefs(self, mock_load, mock_save, ctx):
        """Setting only model preserves existing execution_mode."""
        mock_load.return_value = {"execution_mode": "yolo", "model": None}
        await set_preferences(model="test-model", ctx=ctx)
        mock_save.assert_awaited_once_with(
            ctx,
            {**_ALL_NONE_PREFS, "execution_mode": "yolo", "model": "test-model"},
        )

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_all_none_params_preserves_existing(self, mock_load, mock_save, ctx):
        """set_preferences() with no args keeps existing state intact."""
        mock_load.return_value = {"execution_mode": "yolo", "model": "test-model"}
        await set_preferences(ctx=ctx)
        mock_save.assert_awaited_once_with(
            ctx,
            {**_ALL_NONE_PREFS, "execution_mode": "yolo", "model": "test-model"},
        )

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_returns_json_formatted_confirmation(self, mock_load, mock_save, ctx):
        """Return string contains JSON (null not None, double-quoted keys)."""
        import json

        mock_load.return_value = None
        result = await set_preferences(execution_mode="default", ctx=ctx)
        assert isinstance(result, str)
        # Extract the JSON portion after "Preferences set: "
        json_part = result.split("Preferences set: ", 1)[1]
        parsed = json.loads(json_part)  # raises if not valid JSON
        assert parsed["execution_mode"] == "default"

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_execution_mode_resets_to_none(self, mock_load, mock_save, ctx):
        """clear_execution_mode=True resets execution_mode while keeping model."""
        mock_load.return_value = {"execution_mode": "yolo", "model": "test-model"}
        await set_preferences(clear_execution_mode=True, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "model": "test-model"})

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_model_resets_to_none(self, mock_load, mock_save, ctx):
        """clear_model=True resets model while keeping execution_mode."""
        mock_load.return_value = {"execution_mode": "yolo", "model": "test-model"}
        await set_preferences(clear_model=True, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "execution_mode": "yolo"})

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_flag_takes_precedence_over_value(self, mock_load, mock_save, ctx):
        """clear_execution_mode=True ignores any passed execution_mode value."""
        mock_load.return_value = None
        await set_preferences(execution_mode="yolo", clear_execution_mode=True, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, _ALL_NONE_PREFS)

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_sets_max_retries(self, mock_load, mock_save, ctx):
        """set_preferences(max_retries=5) stores max_retries."""
        mock_load.return_value = None
        await set_preferences(max_retries=5, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "max_retries": 5})

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_sets_output_limit(self, mock_load, mock_save, ctx):
        """set_preferences(output_limit=4096) stores output_limit."""
        mock_load.return_value = None
        await set_preferences(output_limit=4096, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "output_limit": 4096})

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_sets_timeout(self, mock_load, mock_save, ctx):
        """set_preferences(timeout=30) stores timeout."""
        mock_load.return_value = None
        await set_preferences(timeout=30, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "timeout": 30})

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_max_retries(self, mock_load, mock_save, ctx):
        """clear_max_retries=True resets max_retries to None."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": 5,
            "output_limit": None,
            "timeout": None,
        }
        await set_preferences(clear_max_retries=True, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, _ALL_NONE_PREFS)

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_output_limit(self, mock_load, mock_save, ctx):
        """clear_output_limit=True resets output_limit to None."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": 4096,
            "timeout": None,
        }
        await set_preferences(clear_output_limit=True, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, _ALL_NONE_PREFS)

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_timeout(self, mock_load, mock_save, ctx):
        """clear_timeout=True resets timeout to None."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": None,
            "timeout": 30,
        }
        await set_preferences(clear_timeout=True, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, _ALL_NONE_PREFS)

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_returns_confirmation_string(self, mock_load, mock_save, ctx):
        mock_load.return_value = None
        result = await set_preferences(execution_mode="default", ctx=ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_sets_retry_base_delay(self, mock_load, mock_save, ctx):
        """set_preferences(retry_base_delay=1.5) stores retry_base_delay."""
        mock_load.return_value = None
        await set_preferences(retry_base_delay=1.5, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "retry_base_delay": 1.5})

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_sets_retry_max_delay(self, mock_load, mock_save, ctx):
        """set_preferences(retry_max_delay=60.0) stores retry_max_delay."""
        mock_load.return_value = None
        await set_preferences(retry_max_delay=60.0, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, {**_ALL_NONE_PREFS, "retry_max_delay": 60.0})

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_retry_base_delay(self, mock_load, mock_save, ctx):
        """clear_retry_base_delay=True resets retry_base_delay to None."""
        mock_load.return_value = {**_ALL_NONE_PREFS, "retry_base_delay": 1.5}
        await set_preferences(clear_retry_base_delay=True, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, _ALL_NONE_PREFS)

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_retry_max_delay(self, mock_load, mock_save, ctx):
        """clear_retry_max_delay=True resets retry_max_delay to None."""
        mock_load.return_value = {**_ALL_NONE_PREFS, "retry_max_delay": 60.0}
        await set_preferences(clear_retry_max_delay=True, ctx=ctx)
        mock_save.assert_awaited_once_with(ctx, _ALL_NONE_PREFS)


# ---------------------------------------------------------------------------
# TestGetPreferences
# ---------------------------------------------------------------------------


class TestGetPreferences:
    async def test_raises_tool_error_when_ctx_none(self):
        with pytest.raises(ToolError):
            await get_preferences(ctx=None)

    @patch(_LOAD)
    async def test_returns_prefs_dict(self, mock_load, ctx):
        mock_load.return_value = {"execution_mode": "yolo", "model": "test-model"}
        result = await get_preferences(ctx=ctx)
        assert result == {
            **_ALL_NONE_PREFS,
            "execution_mode": "yolo",
            "model": "test-model",
        }

    @patch(_LOAD)
    async def test_returns_empty_defaults_when_no_prefs(self, mock_load, ctx):
        mock_load.return_value = None
        result = await get_preferences(ctx=ctx)
        assert result == _ALL_NONE_PREFS

    @patch(_LOAD)
    async def test_returns_dict_not_pydantic(self, mock_load, ctx):
        mock_load.return_value = None
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

    @patch(_DELETE)
    async def test_calls_delete_preferences(self, mock_delete, ctx):
        result = await clear_preferences(ctx=ctx)
        mock_delete.assert_awaited_once_with(ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch(_DELETE)
    async def test_returns_confirmation_string(self, mock_delete, ctx):
        result = await clear_preferences(ctx=ctx)
        assert "cleared" in result.lower()


# ---------------------------------------------------------------------------
# TestPromptPreferenceFallback
# ---------------------------------------------------------------------------


class TestPromptPreferenceFallback:
    @patch(_LOAD)
    async def test_session_mode_used_when_no_explicit_mode(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent execution_mode='yolo' is used when prompt() called without explicit mode."""
        mock_load.return_value = {"execution_mode": "yolo", "model": None}

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.sandbox == "danger_full_access"
        assert config.approval_policy == "never"

    @patch(_LOAD)
    async def test_explicit_mode_overrides_session(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Explicit execution_mode='default' overrides persistent 'yolo'."""
        mock_load.return_value = {"execution_mode": "yolo", "model": None}

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", execution_mode="default", ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config == ExecutionConfigValues()

    @patch(_LOAD)
    async def test_session_model_used_when_no_explicit_model(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent model is used when prompt() called without explicit model."""
        mock_load.return_value = {"execution_mode": None, "model": "test-model"}

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.model == "test-model"

    @patch(_LOAD)
    async def test_explicit_model_overrides_session(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Explicit model overrides persistent model."""
        mock_load.return_value = {"execution_mode": None, "model": "test-model"}

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", model="alternate-model", ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.model == "alternate-model"

    async def test_no_session_no_explicit_uses_defaults(self, preference_job_service_boundary):
        """Without persistent prefs or explicit params, execution_mode='default' and model=None."""
        await prompt(cli=REPRESENTATIVE_CLI, prompt="test")  # ctx=None → no prefs

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config == ExecutionConfigValues()

    async def test_ctx_none_does_not_crash(self, preference_job_service_boundary):
        """prompt() with ctx=None falls back to defaults without raising."""
        result = await prompt(cli=REPRESENTATIVE_CLI, prompt="test", ctx=None)

        assert strip_runner_header(result) == "ok"
        assert (
            preference_job_service_boundary.only_config_for_prompt("test")
            == ExecutionConfigValues()
        )

    @patch(_LOAD)
    async def test_session_max_retries_used_when_no_explicit(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent max_retries is used when prompt() called without explicit max_retries."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": 5,
            "output_limit": None,
            "timeout": None,
        }

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", ctx=ctx)

        retry_policy = preference_job_service_boundary.only_config_for_prompt("test").retry_policy
        assert retry_policy is not None
        assert retry_policy.max_attempts == 5

    @patch(_LOAD)
    async def test_explicit_max_retries_overrides_session(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Explicit max_retries overrides persistent max_retries."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": 5,
            "output_limit": None,
            "timeout": None,
        }

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", max_retries=2, ctx=ctx)

        retry_policy = preference_job_service_boundary.only_config_for_prompt("test").retry_policy
        assert retry_policy is not None
        assert retry_policy.max_attempts == 2

    @patch(_LOAD)
    async def test_session_output_limit_used_when_no_explicit(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent output_limit is used when prompt() called without explicit output_limit."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": 4096,
            "timeout": None,
        }

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.output_limit_bytes == 4096

    @patch(_LOAD)
    async def test_session_timeout_used_when_no_explicit(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent timeout is used when prompt() called without explicit timeout."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": None,
            "timeout": 30,
        }

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.timeout_seconds == 30

    @patch(_LOAD)
    async def test_session_retry_base_delay_used_when_no_explicit(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent retry_base_delay is captured in typed job configuration."""
        mock_load.return_value = {**_ALL_NONE_PREFS, "retry_base_delay": 1.5}

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", ctx=ctx)

        retry_policy = preference_job_service_boundary.only_config_for_prompt("test").retry_policy
        assert retry_policy is not None
        assert retry_policy.base_delay_seconds == 1.5

    @patch(_LOAD)
    async def test_session_retry_max_delay_used_when_no_explicit(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent retry_max_delay is captured in typed job configuration."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": None,
            "timeout": None,
            "retry_base_delay": None,
            "retry_max_delay": 120.0,
        }

        await prompt(cli=REPRESENTATIVE_CLI, prompt="test", ctx=ctx)

        retry_policy = preference_job_service_boundary.only_config_for_prompt("test").retry_policy
        assert retry_policy is not None
        assert retry_policy.max_delay_seconds == 120.0


# ---------------------------------------------------------------------------
# TestBatchPromptPreferenceFallback
# ---------------------------------------------------------------------------


class TestBatchPromptPreferenceFallback:
    @patch(_LOAD)
    async def test_session_mode_applied_to_tasks_with_none_mode(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent execution_mode='yolo' is applied to tasks with execution_mode=None."""
        mock_load.return_value = {"execution_mode": "yolo", "model": None}

        tasks = [AgentTask(cli=REPRESENTATIVE_CLI, prompt="test", execution_mode=None)]
        await batch_prompt(tasks=tasks, ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.sandbox == "danger_full_access"
        assert config.approval_policy == "never"

    @patch(_LOAD)
    async def test_explicit_task_mode_not_overridden(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Task with explicit execution_mode='default' keeps it despite persistent 'yolo'."""
        mock_load.return_value = {"execution_mode": "yolo", "model": None}

        tasks = [AgentTask(cli=REPRESENTATIVE_CLI, prompt="test", execution_mode="default")]
        await batch_prompt(tasks=tasks, ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config == ExecutionConfigValues()

    @patch(_LOAD)
    async def test_session_model_applied_to_tasks(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent model is applied to tasks with model=None."""
        mock_load.return_value = {"execution_mode": None, "model": "test-model"}

        tasks = [AgentTask(cli=REPRESENTATIVE_CLI, prompt="test")]
        await batch_prompt(tasks=tasks, ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.model == "test-model"

    async def test_no_session_resolves_none_to_default(self, preference_job_service_boundary):
        """Without persistent prefs, task with execution_mode=None resolves to 'default'."""
        tasks = [AgentTask(cli=REPRESENTATIVE_CLI, prompt="test", execution_mode=None)]
        await batch_prompt(tasks=tasks)  # ctx=None

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config == ExecutionConfigValues()

    @patch(_LOAD)
    async def test_session_max_retries_applied_to_tasks(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent max_retries is applied to tasks with max_retries=None."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": 5,
            "output_limit": None,
            "timeout": None,
        }

        tasks = [AgentTask(cli=REPRESENTATIVE_CLI, prompt="test")]
        await batch_prompt(tasks=tasks, ctx=ctx)

        retry_policy = preference_job_service_boundary.only_config_for_prompt("test").retry_policy
        assert retry_policy is not None
        assert retry_policy.max_attempts == 5

    @patch(_LOAD)
    async def test_session_output_limit_applied_to_tasks(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent output_limit is applied to tasks with output_limit=None."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": 4096,
            "timeout": None,
        }

        tasks = [AgentTask(cli=REPRESENTATIVE_CLI, prompt="test")]
        await batch_prompt(tasks=tasks, ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.output_limit_bytes == 4096

    @patch(_LOAD)
    async def test_session_timeout_applied_to_tasks(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent timeout is applied to tasks with timeout=None."""
        mock_load.return_value = {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": None,
            "timeout": 30,
        }

        tasks = [AgentTask(cli=REPRESENTATIVE_CLI, prompt="test")]
        await batch_prompt(tasks=tasks, ctx=ctx)

        config = preference_job_service_boundary.only_config_for_prompt("test")
        assert config.timeout_seconds == 30

    @patch(_LOAD)
    async def test_mixed_tasks_selective_apply(
        self, mock_load, preference_job_service_boundary, ctx
    ):
        """Persistent mode applies to None-mode tasks; explicit-mode tasks keep theirs."""
        mock_load.return_value = {"execution_mode": "yolo", "model": None}

        tasks = [
            AgentTask(cli=REPRESENTATIVE_CLI, prompt="t1", execution_mode=None),
            AgentTask(cli=REPRESENTATIVE_CLI, prompt="t2", execution_mode="default"),
        ]
        await batch_prompt(tasks=tasks, ctx=ctx)

        assert preference_job_service_boundary.prompt_order() == ["t1", "t2"]
        yolo_config = preference_job_service_boundary.config_for_prompt("t1")
        assert yolo_config.sandbox == "danger_full_access"
        assert yolo_config.approval_policy == "never"
        assert preference_job_service_boundary.config_for_prompt("t2") == ExecutionConfigValues()


# ---------------------------------------------------------------------------
# Task 10: TestElicitationPreferences
# ---------------------------------------------------------------------------


class TestElicitationPreferences:
    @patch(_SAVE)
    @patch(_LOAD)
    async def test_set_elicit_false(self, mock_load, mock_save, ctx):
        """set_preferences(elicit=False) stores elicit=False in persistent store."""
        mock_load.return_value = None

        result = await set_preferences(elicit=False, ctx=ctx)

        assert '"elicit": false' in result

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_set_confirm_yolo_false(self, mock_load, mock_save, ctx):
        """set_preferences(confirm_yolo=False) stores confirm_yolo=False."""
        mock_load.return_value = None

        result = await set_preferences(confirm_yolo=False, ctx=ctx)

        assert '"confirm_yolo": false' in result

    @patch(_SAVE)
    @patch(_LOAD)
    async def test_clear_confirm_yolo(self, mock_load, mock_save, ctx):
        """clear_confirm_yolo=True resets confirm_yolo to null even if existing value is False."""
        mock_load.return_value = {"confirm_yolo": False}

        result = await set_preferences(clear_confirm_yolo=True, ctx=ctx)

        assert '"confirm_yolo": null' in result
