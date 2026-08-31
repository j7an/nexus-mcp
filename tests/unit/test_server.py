# tests/unit/test_server.py
"""Tests for the FastMCP server and tool functions.

Compatibility-tool tests patch the durable job-service boundary. Runner and subprocess behavior is
covered by the pipeline suites, so these tests describe only the MCP adapter contract.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.config import get_tool_timeout
from nexus_mcp.core import (
    ExecutionConfigValues,
    FailedJobResultResponse,
    JobError,
    JobEvent,
    JobResultEnvelope,
    RetryPolicy,
    SucceededJobResultResponse,
    TurnOperation,
    TurnResult,
    WorkspaceSelector,
)
from nexus_mcp.emitters import make_mcp_emitter
from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.jobs import AgentJobService
from nexus_mcp.labels import assign_labels
from nexus_mcp.mcp.runtime import runtime_provider
from nexus_mcp.mcp.server import mcp as implementation_mcp
from nexus_mcp.server import (
    _inject_cli_enum,
    batch_prompt,
    build_server_instructions,
    mcp,
    prompt,
)
from nexus_mcp.types import DEFAULT_MAX_CONCURRENCY, AgentTask, MultiPromptResponse
from tests.fixtures import (
    REPRESENTATIVE_CLI,
    make_agent_task,
    make_job_handle,
    strip_runner_header,
)


def test_root_server_mcp_is_implementation_instance() -> None:
    """The legacy server import exposes the owning FastMCP instance."""
    assert mcp is implementation_mcp


def test_root_server_set_preferences_is_implementation_function() -> None:
    """The legacy server module preserves the set_preferences import."""
    from nexus_mcp.mcp.preferences import set_preferences as implementation_set_preferences
    from nexus_mcp.server import set_preferences as compatibility_set_preferences

    assert compatibility_set_preferences is implementation_set_preferences


def test_root_server_clear_preferences_is_implementation_function() -> None:
    """The legacy server module preserves the clear_preferences import."""
    from nexus_mcp.mcp.preferences import clear_preferences as implementation_clear_preferences
    from nexus_mcp.server import clear_preferences as compatibility_clear_preferences

    assert compatibility_clear_preferences is implementation_clear_preferences


class _JobServiceBoundary:
    """Deterministic durable-service boundary for compatibility adapter tests."""

    def __init__(
        self,
        *,
        outputs: dict[str, str] | None = None,
        failures: dict[str, JobError] | None = None,
    ) -> None:
        self.outputs = outputs or {}
        self.failures = failures or {}
        self.operations: dict[str, TurnOperation] = {}
        self.start = AsyncMock(side_effect=self._start)
        self.result = AsyncMock(side_effect=self._result)
        self.subscribe_events = Mock(side_effect=self._subscribe_events)
        self.active_subscriptions = 0
        self.max_active_subscriptions = 0

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
            self.active_subscriptions += 1
            self.max_active_subscriptions = max(
                self.max_active_subscriptions, self.active_subscriptions
            )
            try:
                await asyncio.sleep(0)
                terminal = (
                    "job_failed"
                    if self.operations[job_id].prompt in self.failures
                    else "job_completed"
                )
                yield JobEvent(job_id=job_id, sequence=1, type=terminal)
            finally:
                self.active_subscriptions -= 1

        return events()

    async def _result(self, **kwargs):
        job_id = kwargs["job_id"]
        prompt_text = self.operations[job_id].prompt
        if error := self.failures.get(prompt_text):
            return FailedJobResultResponse(job_id=job_id, error=error)
        return SucceededJobResultResponse(
            job_id=job_id,
            result=JobResultEnvelope(
                job_id=job_id,
                payload=TurnResult(message=self.outputs.get(prompt_text, f"result-{prompt_text}")),
            ),
        )


@pytest.fixture
def job_service_boundary(monkeypatch) -> _JobServiceBoundary:
    boundary = _JobServiceBoundary()
    boundary.install(monkeypatch)
    return boundary


class TestPrompt:
    """Tests for the prompt tool function."""

    async def test_prompt_submits_turn_and_returns_compatibility_output(
        self, job_service_boundary, fake_runner_registry
    ):
        """prompt submits one cwd-scoped turn and preserves its legacy response header."""
        job_service_boundary.outputs["Test prompt"] = "Agent response"

        result = await prompt(
            cli=REPRESENTATIVE_CLI,
            prompt="Test prompt",
        )

        assert result == "[cli: fake | model: default | mode: default]\n\nAgent response"
        assert strip_runner_header(result) == "Agent response"
        call = job_service_boundary.start.await_args.kwargs
        assert call["workspace"] == WorkspaceSelector(path=Path.cwd())
        assert call["backend_id"] == REPRESENTATIVE_CLI
        assert call["operation"] == TurnOperation(prompt="Test prompt")
        assert call["explicit_config"] == ExecutionConfigValues()
        job_service_boundary.subscribe_events.assert_called_once()
        job_service_boundary.result.assert_awaited_once()

    async def test_prompt_maps_yolo_to_truthful_job_policy(
        self, job_service_boundary, fake_runner_registry
    ):
        """execution_mode=yolo admits danger_full_access plus never approval."""

        await prompt(
            cli=REPRESENTATIVE_CLI,
            prompt="Complex task",
            execution_mode="yolo",
        )

        config = job_service_boundary.start.await_args.kwargs["explicit_config"]
        assert config.sandbox == "danger_full_access"
        assert config.approval_policy == "never"
        assert config.sandbox != "workspace_write"

    async def test_prompt_maps_model_context_and_retry_values(
        self, job_service_boundary, fake_runner_registry
    ):
        """Legacy prompt fields become a typed operation and explicit job configuration."""

        await prompt(
            cli=REPRESENTATIVE_CLI,
            prompt="Test prompt",
            model="representative-model",
            context={"key": "value"},
            max_retries=7,
            output_limit=4096,
            timeout=25,
            retry_base_delay=0.25,
            retry_max_delay=2.0,
        )

        call = job_service_boundary.start.await_args.kwargs
        assert call["operation"] == TurnOperation(prompt="Test prompt", context={"key": "value"})
        config = call["explicit_config"]
        assert config.model == "representative-model"
        assert config.output_limit_bytes == 4096
        assert config.timeout_seconds == 25
        assert config.retry_policy == RetryPolicy(
            max_attempts=7,
            base_delay_seconds=0.25,
            max_delay_seconds=2.0,
        )

    async def test_prompt_formats_live_admission_exception_type(
        self, job_service_boundary, fake_runner_registry
    ):
        """An exception before a job handle keeps its live Python class name."""
        job_service_boundary.start.side_effect = UnsupportedAgentError("unknown_agent")

        with pytest.raises(ToolError, match=r"\[UnsupportedAgentError\].*unknown_agent"):
            await prompt(
                cli=REPRESENTATIVE_CLI,
                prompt="Test prompt",
            )

    async def test_prompt_formats_durable_failure_type_and_domain_message(
        self, monkeypatch, fake_runner_registry
    ):
        """A failed job uses bounded legacy type detail and its stable domain message."""
        failure = JobError(
            code="structured_output_invalid",
            message="Legacy backend fake returned an invalid response",
            details={"legacy_exception_type": "ParseError"},
        )
        boundary = _JobServiceBoundary(failures={"Test prompt": failure})
        boundary.install(monkeypatch)

        with pytest.raises(
            ToolError,
            match=r"\[ParseError\] Legacy backend fake returned an invalid response",
        ):
            await prompt(
                cli=REPRESENTATIVE_CLI,
                prompt="Test prompt",
            )

    async def test_prompt_ignores_invalid_legacy_exception_type(
        self, monkeypatch, fake_runner_registry
    ):
        """Malformed failure details never invent a compatibility exception class."""
        failure = JobError(
            code="provider_failed",
            message="Legacy backend fake process failed",
            details={"legacy_exception_type": "not a valid type!"},
        )
        boundary = _JobServiceBoundary(failures={"Test prompt": failure})
        boundary.install(monkeypatch)

        with pytest.raises(ToolError) as raised:
            await prompt(
                cli=REPRESENTATIVE_CLI,
                prompt="Test prompt",
            )

        assert str(raised.value) == "Legacy backend fake process failed"


class TestToolTimeoutRegistration:
    """Verify that tool-level timeouts are set on registered FunctionTools.

    Timeout is baked in at module import time via get_tool_timeout().
    These tests verify the default (900.0s) is applied to prompt/batch_prompt.
    """

    async def test_prompt_has_timeout(self):
        """prompt tool is registered with the default tool timeout."""
        tool = await mcp.get_tool("prompt")
        assert tool.timeout == get_tool_timeout()

    async def test_batch_prompt_has_timeout(self):
        """batch_prompt tool is registered with the default tool timeout."""
        tool = await mcp.get_tool("batch_prompt")
        assert tool.timeout == get_tool_timeout()


class TestAssignLabels:
    """Tests for the assign_labels() pure helper."""

    def test_single_task_gets_agent_name(self):
        """A single unlabeled task gets its agent name as label."""
        tasks = [make_agent_task(cli=REPRESENTATIVE_CLI)]
        result = assign_labels(tasks)
        assert result[0].label == REPRESENTATIVE_CLI

    def test_two_identical_agents_get_suffixes(self):
        """Two tasks with the same agent get 'agent' and 'agent-2'."""
        tasks = [
            make_agent_task(cli=REPRESENTATIVE_CLI),
            make_agent_task(cli=REPRESENTATIVE_CLI),
        ]
        result = assign_labels(tasks)
        assert result[0].label == REPRESENTATIVE_CLI
        assert result[1].label == f"{REPRESENTATIVE_CLI}-2"

    def test_three_identical_agents_get_suffixes(self):
        """Three tasks with the same agent get 'agent', 'agent-2', 'agent-3'."""
        tasks = [make_agent_task(cli=REPRESENTATIVE_CLI) for _ in range(3)]
        result = assign_labels(tasks)
        assert result[0].label == REPRESENTATIVE_CLI
        assert result[1].label == f"{REPRESENTATIVE_CLI}-2"
        assert result[2].label == f"{REPRESENTATIVE_CLI}-3"

    def test_explicit_label_preserved(self):
        """An explicit label is kept as-is, not overwritten."""
        tasks = [make_agent_task(cli=REPRESENTATIVE_CLI, label="my-task")]
        result = assign_labels(tasks)
        assert result[0].label == "my-task"

    def test_explicit_label_blocks_auto_name(self):
        """If a representative label exists, auto-assigned labels get suffixes."""
        tasks = [
            make_agent_task(cli=REPRESENTATIVE_CLI, label=REPRESENTATIVE_CLI),
            make_agent_task(cli=REPRESENTATIVE_CLI),
        ]
        result = assign_labels(tasks)
        assert result[0].label == REPRESENTATIVE_CLI
        assert result[1].label == f"{REPRESENTATIVE_CLI}-2"

    def test_mixed_agents_no_suffix(self):
        """Different agents don't get suffixes when there are no collisions."""
        tasks = [make_agent_task(cli=REPRESENTATIVE_CLI), make_agent_task(cli="codex")]
        result = assign_labels(tasks)
        assert result[0].label == REPRESENTATIVE_CLI
        assert result[1].label == "codex"

    def test_returns_new_list_does_not_mutate(self):
        """assign_labels() returns a new list; input tasks are unchanged."""
        tasks = [make_agent_task(cli=REPRESENTATIVE_CLI)]
        assert tasks[0].label is None
        result = assign_labels(tasks)
        assert tasks[0].label is None  # original unchanged
        assert result is not tasks
        assert result[0] is not tasks[0]

    def test_empty_list_returns_empty(self):
        """An empty input list returns an empty list."""
        assert assign_labels([]) == []


class TestBatchPrompt:
    """Tests for the batch_prompt tool function."""

    async def test_successes_are_independent_jobs_in_input_order(
        self, job_service_boundary, fake_runner_registry
    ):
        """Each batch entry gets its own session/job while result order stays stable."""
        tasks = [make_agent_task(prompt=f"p{i}") for i in range(3)]

        result = await batch_prompt(tasks=tasks)

        assert isinstance(result, MultiPromptResponse)
        assert result.succeeded == 3
        assert result.failed == 0
        assert [strip_runner_header(item.output or "") for item in result.results] == [
            "result-p0",
            "result-p1",
            "result-p2",
        ]
        assert job_service_boundary.start.await_count == 3
        assert len(job_service_boundary.operations) == 3
        prompts = {
            call.kwargs["operation"].prompt for call in job_service_boundary.start.await_args_list
        }
        assert prompts == {
            "p0",
            "p1",
            "p2",
        }

    async def test_partial_durable_failure_preserves_good_result_and_type(
        self, monkeypatch, fake_runner_registry
    ):
        """One failed durable result does not discard a sibling success."""
        boundary = _JobServiceBoundary(
            outputs={"ok": "good output"},
            failures={
                "bad": JobError(
                    code="structured_output_invalid",
                    message="Legacy backend fake returned an invalid response",
                    details={"legacy_exception_type": "ParseError"},
                )
            },
        )
        boundary.install(monkeypatch)

        result = await batch_prompt(
            tasks=[make_agent_task(prompt="ok"), make_agent_task(prompt="bad")]
        )

        assert result.succeeded == 1
        assert result.failed == 1
        assert strip_runner_header(result.results[0].output or "") == "good output"
        assert result.results[1].error == "Legacy backend fake returned an invalid response"
        assert result.results[1].error_type == "ParseError"

    async def test_max_concurrency_limits_active_job_waiters(
        self, job_service_boundary, fake_runner_registry
    ):
        """The adapter semaphore bounds independently durable job waiters."""
        await batch_prompt(
            tasks=[make_agent_task(prompt=f"p{i}") for i in range(5)],
            max_concurrency=2,
        )

        assert job_service_boundary.max_active_subscriptions == 2

    async def test_labels_auto_assigned(self, job_service_boundary, fake_runner_registry):
        """Unlabeled tasks receive unique auto-assigned labels."""
        result = await batch_prompt(
            tasks=[
                make_agent_task(cli=REPRESENTATIVE_CLI),
                make_agent_task(cli=REPRESENTATIVE_CLI),
            ]
        )

        assert [item.label for item in result.results] == ["fake", "fake-2"]

    async def test_empty_task_list(self, job_service_boundary, monkeypatch):
        """An empty task list returns immediately without borrowing a job runtime."""
        borrow = Mock(side_effect=AssertionError("empty batch must not borrow a runtime"))
        monkeypatch.setattr(runtime_provider, "borrow", borrow)

        result = await batch_prompt(tasks=[])

        assert result.total == 0
        assert result.results == []
        job_service_boundary.start.assert_not_awaited()
        borrow.assert_not_called()

    def test_default_concurrency_is_three(self):
        """DEFAULT_MAX_CONCURRENCY constant equals 3."""
        assert DEFAULT_MAX_CONCURRENCY == 3

    async def test_max_concurrency_zero_raises_value_error(self):
        """max_concurrency=0 raises ValueError before creating a deadlocking Semaphore(0)."""
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await batch_prompt(tasks=[make_agent_task()], max_concurrency=0)

    async def test_max_concurrency_negative_raises_value_error(self):
        """max_concurrency=-1 raises ValueError."""
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await batch_prompt(tasks=[make_agent_task()], max_concurrency=-1)

    async def test_single_task_no_suffix(self, job_service_boundary, fake_runner_registry):
        """A single task's label is the agent name without any suffix."""
        result = await batch_prompt(tasks=[make_agent_task(cli=REPRESENTATIVE_CLI)])

        assert result.results[0].label == REPRESENTATIVE_CLI

    async def test_ctx_info_called_on_start_and_complete(
        self, job_service_boundary, fake_runner_registry, ctx
    ):
        """ctx.info() is awaited exactly twice: once at start, once at completion."""
        await batch_prompt(tasks=[make_agent_task()], ctx=ctx)

        assert ctx.info.await_count == 2

    async def test_ctx_none_does_not_raise(self, job_service_boundary, fake_runner_registry):
        """Default ctx=None completes without error (documents the None contract)."""
        result = await batch_prompt(tasks=[make_agent_task()])
        assert isinstance(result, MultiPromptResponse)


class TestServerInstructions:
    """Tests for the build_server_instructions() function."""

    def test_instructions_is_non_empty_string(self):
        """build_server_instructions() returns a non-empty markdown string."""
        result = build_server_instructions()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_instructions_mention_all_runner_names(self):
        """Instructions contain all registered runner names."""
        result = build_server_instructions()
        for name in ("claude", "codex", "opencode", "opencode_server"):
            assert name in result

    def test_instructions_include_availability_status(self):
        """Instructions contain availability markers for each runner."""
        result = build_server_instructions()
        # At minimum, some runners should show installed/not found status
        assert "installed" in result or "not found" in result

    def test_instructions_include_execution_modes(self):
        """Instructions mention execution modes for runners."""
        result = build_server_instructions()
        assert "default" in result
        assert "yolo" in result

    def test_instructions_include_models_when_set(self, monkeypatch):
        """When NEXUS_CODEX_MODELS is set, instructions list those models."""
        monkeypatch.setenv("NEXUS_CODEX_MODELS", "gpt-5.4-mini,gpt-5.3-codex")
        result = build_server_instructions()
        assert "gpt-5.4-mini" in result
        assert "gpt-5.3-codex" in result

    def test_instructions_include_default_model_when_configured(self, monkeypatch):
        """When NEXUS_{RUNNER}_MODEL is set, instructions include the default model line."""
        monkeypatch.setenv("NEXUS_CODEX_MODEL", "gpt-5.4")
        result = build_server_instructions()
        assert "- Default model: gpt-5.4" in result


class TestInjectCliEnumEdgeCases:
    """Tests for _inject_cli_enum() defensive closure branches (lines 95-98)."""

    def test_chains_existing_callable_json_schema_extra(self, monkeypatch, request):
        """When AgentTask already has a callable json_schema_extra, it is invoked."""
        extra_was_called = []

        def existing_extra(schema: dict) -> None:
            extra_was_called.append(True)

        monkeypatch.setitem(AgentTask.model_config, "json_schema_extra", existing_extra)
        request.addfinalizer(lambda: AgentTask.model_rebuild(force=True))

        _inject_cli_enum()
        schema = AgentTask.model_json_schema()

        assert extra_was_called, "Pre-existing callable extra was not invoked"
        assert "enum" in schema.get("properties", {}).get("cli", {})

    def test_merges_existing_dict_json_schema_extra(self, monkeypatch, request):
        """When AgentTask already has a dict json_schema_extra, it is merged."""
        monkeypatch.setitem(AgentTask.model_config, "json_schema_extra", {"x-custom": "preserved"})
        request.addfinalizer(lambda: AgentTask.model_rebuild(force=True))

        _inject_cli_enum()
        schema = AgentTask.model_json_schema()

        assert schema.get("x-custom") == "preserved"
        assert "enum" in schema.get("properties", {}).get("cli", {})


class TestDynamicCliEnum:
    """Tests for dynamic CLI enum injection into tool schemas."""

    async def test_prompt_schema_has_cli_enum(self):
        """prompt tool's cli parameter has an enum listing all runner names."""
        tool = await mcp.get_tool("prompt")
        cli_schema = tool.parameters["properties"]["cli"]
        assert "enum" in cli_schema
        assert set(cli_schema["enum"]) == {
            "claude",
            "codex",
            "opencode",
            "opencode_server",
        }

    async def test_batch_prompt_task_cli_has_enum(self):
        """batch_prompt's task schema has cli enum in the nested AgentTask definition."""
        tool = await mcp.get_tool("batch_prompt")
        # Navigate: properties → tasks → items → properties → cli
        tasks_schema = tool.parameters["properties"]["tasks"]
        # items may be under "items" directly or in "$defs"
        items = tasks_schema.get("items", {})
        # For Pydantic models, items may reference $defs — resolve if needed
        if "$ref" in items:
            ref_name = items["$ref"].split("/")[-1]
            items = tool.parameters.get("$defs", {}).get(ref_name, {})
        cli_field = items.get("properties", {}).get("cli", {})
        assert "enum" in cli_field
        assert set(cli_field["enum"]) == {
            "claude",
            "codex",
            "opencode",
            "opencode_server",
        }

    async def test_instructions_are_set_on_mcp(self):
        """FastMCP server has non-empty instructions after module load."""
        assert mcp.instructions is not None
        assert len(mcp.instructions) > 0
        assert "nexus-mcp" in mcp.instructions


class TestMakeMcpEmitter:
    """make_mcp_emitter creates a dual-output emitter."""

    async def test_info_calls_both_ctx_and_logger(self, ctx):
        """Info level calls ctx.info() and logger.info()."""
        emitter = make_mcp_emitter(ctx)

        with patch("nexus_mcp.mcp.emitters.logger") as mock_logger:
            await emitter("info", "test message")

        ctx.info.assert_awaited_once_with("test message")
        mock_logger.info.assert_called_once_with("test message")

    async def test_warning_calls_both_ctx_and_logger(self, ctx):
        """Warning level calls ctx.warning() and logger.warning()."""
        emitter = make_mcp_emitter(ctx)

        with patch("nexus_mcp.mcp.emitters.logger") as mock_logger:
            await emitter("warning", "retry warning")

        ctx.warning.assert_awaited_once_with("retry warning")
        mock_logger.warning.assert_called_once_with("retry warning")

    async def test_error_calls_ctx_and_logger_with_exc_info(self, ctx):
        """Error level calls ctx.error() and logger.error(exc_info=True)."""
        emitter = make_mcp_emitter(ctx)

        with patch("nexus_mcp.mcp.emitters.logger") as mock_logger:
            await emitter("error", "task failed")

        ctx.error.assert_awaited_once_with("task failed")
        mock_logger.error.assert_called_once_with("task failed", exc_info=True)

    async def test_debug_calls_both_ctx_and_logger(self, ctx):
        """Debug level calls ctx.debug() and logger.debug()."""
        emitter = make_mcp_emitter(ctx)

        with patch("nexus_mcp.mcp.emitters.logger") as mock_logger:
            await emitter("debug", "debug message")

        ctx.debug.assert_awaited_once_with("debug message")
        mock_logger.debug.assert_called_once_with("debug message")


class TestPromptElicitation:
    async def test_prompt_accepts_elicit_parameter(
        self, fake_runner_registry, ctx, job_service_boundary
    ):
        job_service_boundary.outputs["Hello world test prompt"] = "fake output"
        result = await prompt(
            cli=fake_runner_registry, prompt="Hello world test prompt", elicit=False, ctx=ctx
        )
        assert "fake output" in result
