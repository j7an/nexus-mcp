# tests/unit/test_server.py
"""Tests for the FastMCP server and tool functions.

Mocking strategy: Mock at the RunnerFactory/runner boundary, NOT subprocess level.
Server tests should be decoupled from runner internals.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.config import get_tool_timeout
from nexus_mcp.exceptions import (
    ParseError,
    SubprocessError,
    UnsupportedAgentError,
)
from nexus_mcp.server import (
    _assign_labels,
    _inject_cli_enum,
    batch_prompt,
    build_server_instructions,
    mcp,
    prompt,
)
from nexus_mcp.types import DEFAULT_MAX_CONCURRENCY, AgentTask, MultiPromptResponse
from tests.fixtures import make_agent_response, make_agent_task


def _setup_mock_runner(mock_factory, *, output: str = "test output", side_effect=None) -> AsyncMock:
    """Configure mock_factory.create() to return a runner with preset run() behavior.

    Args:
        mock_factory: The patched RunnerFactory mock.
        output: The output string for run.return_value (used when side_effect is None).
        side_effect: If set, assigned to run.side_effect instead of return_value.

    Returns:
        The configured AsyncMock runner (for further assertion access).
    """
    mock_runner = AsyncMock()
    if side_effect is not None:
        mock_runner.run.side_effect = side_effect
    else:
        mock_runner.run.return_value = make_agent_response(output=output)
    mock_factory.create.return_value = mock_runner
    return mock_runner


class TestPrompt:
    """Tests for the prompt tool function."""

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_returns_response(self, mock_factory):
        """prompt dispatches to runner via batch_prompt and returns output text."""
        mock_runner = _setup_mock_runner(mock_factory, output="Agent response")

        result = await prompt(
            cli="gemini",
            prompt="Test prompt",
        )

        assert result == "Agent response"
        mock_factory.create.assert_called_once_with("gemini")
        call_args = mock_runner.run.call_args.args[0]
        assert call_args.prompt == "Test prompt"
        assert call_args.context == {}
        assert call_args.execution_mode == "default"
        assert call_args.model is None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_passes_execution_mode(self, mock_factory):
        """execution_mode is passed through to the PromptRequest."""
        mock_runner = _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            cli="gemini",
            prompt="Complex task",
            execution_mode="yolo",
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "yolo"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_passes_model(self, mock_factory):
        """model parameter is passed through to the PromptRequest."""
        mock_runner = _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            cli="gemini",
            prompt="Test prompt",
            model="gemini-2.5-flash",
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.model == "gemini-2.5-flash"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_passes_context(self, mock_factory):
        """context is passed through to the PromptRequest."""
        mock_runner = _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            cli="gemini",
            prompt="Test prompt",
            context={"key": "value"},
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.context == {"key": "value"}

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_handles_unsupported_agent(self, mock_factory):
        """ToolError raised when factory cannot create runner for unknown agent."""
        mock_factory.create.side_effect = UnsupportedAgentError("unknown_agent")

        with pytest.raises(ToolError, match="unknown_agent"):
            await prompt(
                cli="unknown_agent",
                prompt="Test prompt",
            )

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_handles_subprocess_error(self, mock_factory):
        """ToolError raised when runner.run() fails."""
        _setup_mock_runner(
            mock_factory,
            side_effect=SubprocessError("CLI command failed", stderr="error output", returncode=1),
        )

        with pytest.raises(ToolError, match="CLI command failed"):
            await prompt(
                cli="gemini",
                prompt="Test prompt",
            )

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_passes_max_retries(self, mock_factory):
        """max_retries parameter is passed through to the PromptRequest."""
        mock_runner = _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            cli="gemini",
            prompt="Test prompt",
            max_retries=7,
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.max_retries == 7

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_max_retries_defaults_to_none(self, mock_factory):
        """max_retries defaults to None when not specified in prompt()."""
        mock_runner = _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            cli="gemini",
            prompt="Test prompt",
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.max_retries is None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_ctx_forwarded_to_batch_prompt(self, mock_factory, ctx):
        """ctx passed to prompt() is forwarded through to batch_prompt()."""
        _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            cli="gemini",
            prompt="Test prompt",
            ctx=ctx,
        )

        # batch_prompt calls ctx.info() twice — once at start, once at completion
        assert ctx.info.await_count == 2

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_raises_tool_error_with_type_prefix(self, mock_factory):
        """ToolError message includes [ErrorType] prefix when error_type is set."""
        _setup_mock_runner(
            mock_factory,
            side_effect=ParseError("Invalid JSON from Gemini CLI"),
        )

        with pytest.raises(ToolError, match=r"\[ParseError\].*Invalid JSON from Gemini CLI"):
            await prompt(
                cli="gemini",
                prompt="Test prompt",
            )


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
    """Tests for the _assign_labels() pure helper."""

    def test_single_task_gets_agent_name(self):
        """A single unlabeled task gets its agent name as label."""
        tasks = [make_agent_task(cli="gemini")]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"

    def test_two_identical_agents_get_suffixes(self):
        """Two tasks with the same agent get 'agent' and 'agent-2'."""
        tasks = [make_agent_task(cli="gemini"), make_agent_task(cli="gemini")]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"
        assert result[1].label == "gemini-2"

    def test_three_identical_agents_get_suffixes(self):
        """Three tasks with the same agent get 'agent', 'agent-2', 'agent-3'."""
        tasks = [make_agent_task(cli="gemini") for _ in range(3)]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"
        assert result[1].label == "gemini-2"
        assert result[2].label == "gemini-3"

    def test_explicit_label_preserved(self):
        """An explicit label is kept as-is, not overwritten."""
        tasks = [make_agent_task(cli="gemini", label="my-task")]
        result = _assign_labels(tasks)
        assert result[0].label == "my-task"

    def test_explicit_label_blocks_auto_name(self):
        """If 'gemini' is already an explicit label, auto-assigned gets 'gemini-2'."""
        tasks = [
            make_agent_task(cli="gemini", label="gemini"),
            make_agent_task(cli="gemini"),
        ]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"
        assert result[1].label == "gemini-2"

    def test_mixed_agents_no_suffix(self):
        """Different agents don't get suffixes when there are no collisions."""
        tasks = [make_agent_task(cli="gemini"), make_agent_task(cli="codex")]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"
        assert result[1].label == "codex"

    def test_returns_new_list_does_not_mutate(self):
        """_assign_labels() returns a new list; input tasks are unchanged."""
        tasks = [make_agent_task(cli="gemini")]
        assert tasks[0].label is None
        result = _assign_labels(tasks)
        assert tasks[0].label is None  # original unchanged
        assert result is not tasks
        assert result[0] is not tasks[0]

    def test_empty_list_returns_empty(self):
        """An empty input list returns an empty list."""
        assert _assign_labels([]) == []


class TestBatchPrompt:
    """Tests for the batch_prompt tool function."""

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_all_success(self, mock_factory):
        """All tasks succeed → succeeded=2, failed=0."""
        _setup_mock_runner(mock_factory, output="ok")

        tasks = [make_agent_task(), make_agent_task(prompt="Second")]
        result = await batch_prompt(tasks=tasks)

        assert result.succeeded == 2
        assert result.failed == 0
        assert result.total == 2

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_partial_failure(self, mock_factory):
        """One task ok, one errors → succeeded=1, failed=1, good result preserved."""

        async def run_side_effect(request):
            if request.prompt == "ok":
                return make_agent_response(output="good output")
            raise RuntimeError("agent exploded")

        _setup_mock_runner(mock_factory, side_effect=run_side_effect)

        tasks = [make_agent_task(prompt="ok"), make_agent_task(prompt="bad")]
        result = await batch_prompt(tasks=tasks)

        assert result.succeeded == 1
        assert result.failed == 1
        ok_result = next(r for r in result.results if r.output == "good output")
        assert ok_result is not None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_all_failures(self, mock_factory):
        """All tasks error → succeeded=0, failed=N."""
        _setup_mock_runner(mock_factory, side_effect=RuntimeError("always fails"))

        tasks = [make_agent_task() for _ in range(3)]
        result = await batch_prompt(tasks=tasks)

        assert result.succeeded == 0
        assert result.failed == 3

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_concurrency_limit(self, mock_factory):
        """Max concurrent invocations does not exceed max_concurrency=2."""
        max_concurrent = 0
        current = 0

        async def slow_run(request):
            nonlocal max_concurrent, current
            current += 1
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0)
            current -= 1
            return make_agent_response()

        _setup_mock_runner(mock_factory, side_effect=slow_run)

        tasks = [make_agent_task() for _ in range(5)]
        await batch_prompt(tasks=tasks, max_concurrency=2)

        assert max_concurrent <= 2

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_result_order_matches_input(self, mock_factory):
        """Results are in the same order as the input tasks."""
        call_order: list[str] = []

        async def ordered_run(request):
            call_order.append(request.prompt)
            return make_agent_response(output=f"result-{request.prompt}")

        _setup_mock_runner(mock_factory, side_effect=ordered_run)

        tasks = [make_agent_task(prompt=f"p{i}") for i in range(3)]
        result = await batch_prompt(tasks=tasks)

        outputs = [r.output for r in result.results]
        assert outputs == ["result-p0", "result-p1", "result-p2"]

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_returns_multi_prompt_response(self, mock_factory):
        """Output is a MultiPromptResponse Pydantic model with a 'results' attribute."""
        _setup_mock_runner(mock_factory)

        result = await batch_prompt(tasks=[make_agent_task()])

        assert isinstance(result, MultiPromptResponse)
        assert hasattr(result, "results")

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_labels_auto_assigned(self, mock_factory):
        """Unlabeled tasks receive unique auto-assigned labels."""
        _setup_mock_runner(mock_factory)

        tasks = [make_agent_task(cli="gemini"), make_agent_task(cli="gemini")]
        result = await batch_prompt(tasks=tasks)

        labels = [r.label for r in result.results]
        assert len(set(labels)) == 2  # all unique

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_empty_task_list(self, mock_factory):
        """An empty task list returns total=0 and empty results."""
        result = await batch_prompt(tasks=[])

        assert result.total == 0
        assert result.results == []

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_unexpected_exception_captured(self, mock_factory):
        """RuntimeError from runner is captured as task error, not propagated."""
        _setup_mock_runner(mock_factory, side_effect=RuntimeError("unexpected boom"))

        result = await batch_prompt(tasks=[make_agent_task()])

        assert result.failed == 1
        assert "unexpected boom" in result.results[0].error

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

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_single_task_no_suffix(self, mock_factory):
        """A single task's label is the agent name without any suffix."""
        _setup_mock_runner(mock_factory)

        result = await batch_prompt(tasks=[make_agent_task(cli="gemini")])

        assert result.results[0].label == "gemini"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_ctx_info_called_on_start_and_complete(self, mock_factory, ctx):
        """ctx.info() is awaited exactly twice: once at start, once at completion."""
        _setup_mock_runner(mock_factory)

        await batch_prompt(tasks=[make_agent_task()], ctx=ctx)

        assert ctx.info.await_count == 2

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_ctx_none_does_not_raise(self, mock_factory):
        """Default ctx=None completes without error (documents the None contract)."""
        _setup_mock_runner(mock_factory)

        # Should not raise even though ctx is None (the default)
        result = await batch_prompt(tasks=[make_agent_task()])
        assert isinstance(result, MultiPromptResponse)

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_batch_prompt_passes_max_retries_to_request(self, mock_factory):
        """AgentTask.max_retries is threaded through to PromptRequest."""
        mock_runner = _setup_mock_runner(mock_factory)

        tasks = [make_agent_task(max_retries=5)]
        await batch_prompt(tasks=tasks)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.max_retries == 5

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_batch_prompt_max_retries_none_by_default(self, mock_factory):
        """When AgentTask.max_retries is None, PromptRequest.max_retries is also None."""
        mock_runner = _setup_mock_runner(mock_factory)

        tasks = [make_agent_task()]  # max_retries not specified → None
        await batch_prompt(tasks=tasks)

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.max_retries is None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_batch_prompt_preserves_error_type(self, mock_factory):
        """error_type is set to the exception class name when runner raises."""
        _setup_mock_runner(
            mock_factory,
            side_effect=ParseError("Invalid JSON from Gemini CLI"),
        )

        result = await batch_prompt(tasks=[make_agent_task()])

        assert result.failed == 1
        assert result.results[0].error_type == "ParseError"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_batch_prompt_error_type_none_on_success(self, mock_factory):
        """error_type is None for successful tasks."""
        _setup_mock_runner(mock_factory, output="ok")

        result = await batch_prompt(tasks=[make_agent_task()])

        assert result.succeeded == 1
        assert result.results[0].error_type is None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_batch_prompt_logs_exception(self, mock_factory, caplog):
        """logger.exception() is called when a task raises, capturing the traceback."""
        import logging

        _setup_mock_runner(
            mock_factory,
            side_effect=ParseError("Invalid JSON from Gemini CLI"),
        )

        with caplog.at_level(logging.ERROR, logger="nexus_mcp.server"):
            await batch_prompt(tasks=[make_agent_task()])

        assert len(caplog.records) == 1
        assert caplog.records[0].exc_info is not None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_report_progress_called_per_task(self, mock_factory, ctx):
        """ctx.report_progress is awaited exactly N times for N tasks."""
        _setup_mock_runner(mock_factory)

        tasks = [make_agent_task() for _ in range(4)]
        await batch_prompt(tasks=tasks, ctx=ctx)

        assert ctx.report_progress.await_count == 4

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_report_progress_includes_message(self, mock_factory, ctx):
        """Single-task call passes progress=1, total=1, and a formatted message."""
        _setup_mock_runner(mock_factory)

        await batch_prompt(tasks=[make_agent_task(cli="gemini")], ctx=ctx)

        ctx.report_progress.assert_awaited_once_with(
            progress=1,
            total=1,
            message="Completed task 1/1: gemini",
        )

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_report_progress_total_set_correctly(self, mock_factory, ctx):
        """For N tasks, all ctx.report_progress calls pass total=N."""
        _setup_mock_runner(mock_factory)

        tasks = [make_agent_task() for _ in range(3)]
        await batch_prompt(tasks=tasks, ctx=ctx)

        for call in ctx.report_progress.call_args_list:
            assert call.kwargs["total"] == 3

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_ctx_none_skips_progress_reporting(self, mock_factory):
        """batch_prompt(ctx=None) completes without error — no report_progress calls."""
        _setup_mock_runner(mock_factory)

        # Must not raise AttributeError on None.report_progress
        result = await batch_prompt(tasks=[make_agent_task()])
        assert isinstance(result, MultiPromptResponse)

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_report_progress_on_failure(self, mock_factory, ctx):
        """ctx.report_progress is called in finally even when a task errors."""
        _setup_mock_runner(mock_factory, side_effect=RuntimeError("boom"))

        await batch_prompt(tasks=[make_agent_task()], ctx=ctx)

        # Task failed but progress was still reported
        assert ctx.report_progress.await_count == 1


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
        for name in ("claude", "codex", "gemini", "opencode"):
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
        """When NEXUS_GEMINI_MODELS is set, instructions list those models."""
        monkeypatch.setenv("NEXUS_GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-pro")
        result = build_server_instructions()
        assert "gemini-2.5-flash" in result
        assert "gemini-2.5-pro" in result

    def test_instructions_include_default_model_when_configured(self, monkeypatch):
        """When NEXUS_{RUNNER}_MODEL is set, instructions include the default model line."""
        monkeypatch.setenv("NEXUS_GEMINI_MODEL", "gemini-2.5-pro")
        result = build_server_instructions()
        assert "- Default model: gemini-2.5-pro" in result


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
        assert set(cli_schema["enum"]) == {"claude", "codex", "gemini", "opencode"}

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
        assert set(cli_field["enum"]) == {"claude", "codex", "gemini", "opencode"}

    async def test_instructions_are_set_on_mcp(self):
        """FastMCP server has non-empty instructions after module load."""
        assert mcp.instructions is not None
        assert len(mcp.instructions) > 0
        assert "nexus-mcp" in mcp.instructions
