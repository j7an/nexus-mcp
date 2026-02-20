# tests/unit/test_server.py
"""Tests for the FastMCP server and tool functions.

Mocking strategy: Mock at the RunnerFactory/runner boundary, NOT subprocess level.
Server tests should be decoupled from runner internals.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from nexus_mcp.exceptions import SubprocessError, UnsupportedAgentError
from nexus_mcp.server import _assign_labels, batch_prompt, list_agents, prompt
from nexus_mcp.types import DEFAULT_MAX_CONCURRENCY
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
    async def test_prompt_returns_response(self, mock_factory, progress):
        """prompt dispatches to runner via batch_prompt and returns output text."""
        mock_runner = _setup_mock_runner(mock_factory, output="Agent response")

        result = await prompt(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
        )

        assert result == "Agent response"
        mock_factory.create.assert_called_once_with("gemini")
        call_args = mock_runner.run.call_args.args[0]
        assert call_args.prompt == "Test prompt"
        assert call_args.context == {}
        assert call_args.execution_mode == "default"
        assert call_args.model is None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_passes_execution_mode(self, mock_factory, progress):
        """execution_mode is passed through to the PromptRequest."""
        mock_runner = _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            agent="gemini",
            prompt="Complex task",
            progress=progress,
            execution_mode="yolo",
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "yolo"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_passes_model(self, mock_factory, progress):
        """model parameter is passed through to the PromptRequest."""
        mock_runner = _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
            model="gemini-2.5-flash",
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.model == "gemini-2.5-flash"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_passes_context(self, mock_factory, progress):
        """context is passed through to the PromptRequest."""
        mock_runner = _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
            context={"key": "value"},
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.context == {"key": "value"}

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_reports_progress(self, mock_factory, progress):
        """progress.set_total(1) and increment(1) are called via batch_prompt."""
        _setup_mock_runner(mock_factory, output="Done")

        await prompt(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
        )

        progress.set_total.assert_called_once_with(1)
        assert progress.increment.call_count == 1

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_handles_unsupported_agent(self, mock_factory, progress):
        """RuntimeError raised when factory cannot create runner for unknown agent."""
        mock_factory.create.side_effect = UnsupportedAgentError("unknown_agent")

        with pytest.raises(RuntimeError, match="unknown_agent"):
            await prompt(
                agent="unknown_agent",
                prompt="Test prompt",
                progress=progress,
            )

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_handles_subprocess_error(self, mock_factory, progress):
        """RuntimeError raised when runner.run() fails."""
        _setup_mock_runner(
            mock_factory,
            side_effect=SubprocessError("CLI command failed", stderr="error output", returncode=1),
        )

        with pytest.raises(RuntimeError, match="CLI command failed"):
            await prompt(
                agent="gemini",
                prompt="Test prompt",
                progress=progress,
            )


class TestListAgents:
    """Tests for the list_agents tool function."""

    def test_list_agents_returns_supported_agents(self):
        """list_agents returns exactly the supported agent names."""
        agents = list_agents()
        assert agents == ["gemini"]


class TestAssignLabels:
    """Tests for the _assign_labels() pure helper."""

    def test_single_task_gets_agent_name(self):
        """A single unlabeled task gets its agent name as label."""
        tasks = [make_agent_task(agent="gemini")]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"

    def test_two_identical_agents_get_suffixes(self):
        """Two tasks with the same agent get 'agent' and 'agent-2'."""
        tasks = [make_agent_task(agent="gemini"), make_agent_task(agent="gemini")]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"
        assert result[1].label == "gemini-2"

    def test_three_identical_agents_get_suffixes(self):
        """Three tasks with the same agent get 'agent', 'agent-2', 'agent-3'."""
        tasks = [make_agent_task(agent="gemini") for _ in range(3)]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"
        assert result[1].label == "gemini-2"
        assert result[2].label == "gemini-3"

    def test_explicit_label_preserved(self):
        """An explicit label is kept as-is, not overwritten."""
        tasks = [make_agent_task(agent="gemini", label="my-task")]
        result = _assign_labels(tasks)
        assert result[0].label == "my-task"

    def test_explicit_label_blocks_auto_name(self):
        """If 'gemini' is already an explicit label, auto-assigned gets 'gemini-2'."""
        tasks = [
            make_agent_task(agent="gemini", label="gemini"),
            make_agent_task(agent="gemini"),
        ]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"
        assert result[1].label == "gemini-2"

    def test_mixed_agents_no_suffix(self):
        """Different agents don't get suffixes when there are no collisions."""
        tasks = [make_agent_task(agent="gemini"), make_agent_task(agent="codex")]
        result = _assign_labels(tasks)
        assert result[0].label == "gemini"
        assert result[1].label == "codex"

    def test_returns_new_list_does_not_mutate(self):
        """_assign_labels() returns a new list; input tasks are unchanged."""
        tasks = [make_agent_task(agent="gemini")]
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
    async def test_all_success(self, mock_factory, progress):
        """All tasks succeed → succeeded=2, failed=0."""
        _setup_mock_runner(mock_factory, output="ok")

        tasks = [make_agent_task(), make_agent_task(prompt="Second")]
        raw = await batch_prompt(tasks=tasks, progress=progress)
        data = json.loads(raw)

        assert data["succeeded"] == 2
        assert data["failed"] == 0
        assert data["total"] == 2

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_partial_failure(self, mock_factory, progress):
        """One task ok, one errors → succeeded=1, failed=1, good result preserved."""

        async def run_side_effect(request):
            if request.prompt == "ok":
                return make_agent_response(output="good output")
            raise RuntimeError("agent exploded")

        _setup_mock_runner(mock_factory, side_effect=run_side_effect)

        tasks = [make_agent_task(prompt="ok"), make_agent_task(prompt="bad")]
        raw = await batch_prompt(tasks=tasks, progress=progress)
        data = json.loads(raw)

        assert data["succeeded"] == 1
        assert data["failed"] == 1
        ok_result = next(r for r in data["results"] if r.get("output") == "good output")
        assert ok_result is not None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_all_failures(self, mock_factory, progress):
        """All tasks error → succeeded=0, failed=N."""
        _setup_mock_runner(mock_factory, side_effect=RuntimeError("always fails"))

        tasks = [make_agent_task() for _ in range(3)]
        raw = await batch_prompt(tasks=tasks, progress=progress)
        data = json.loads(raw)

        assert data["succeeded"] == 0
        assert data["failed"] == 3

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_concurrency_limit(self, mock_factory, progress):
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
        await batch_prompt(tasks=tasks, progress=progress, max_concurrency=2)

        assert max_concurrent <= 2

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_progress_called_per_task(self, mock_factory, progress):
        """progress.increment() is called exactly once per task."""
        _setup_mock_runner(mock_factory)

        tasks = [make_agent_task() for _ in range(4)]
        await batch_prompt(tasks=tasks, progress=progress)

        assert progress.increment.call_count == 4

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_result_order_matches_input(self, mock_factory, progress):
        """Results are in the same order as the input tasks."""
        call_order: list[str] = []

        async def ordered_run(request):
            call_order.append(request.prompt)
            return make_agent_response(output=f"result-{request.prompt}")

        _setup_mock_runner(mock_factory, side_effect=ordered_run)

        tasks = [make_agent_task(prompt=f"p{i}") for i in range(3)]
        raw = await batch_prompt(tasks=tasks, progress=progress)
        data = json.loads(raw)

        outputs = [r["output"] for r in data["results"]]
        assert outputs == ["result-p0", "result-p1", "result-p2"]

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_returns_json_string(self, mock_factory, progress):
        """Output is a valid JSON string with a 'results' key."""
        _setup_mock_runner(mock_factory)

        raw = await batch_prompt(tasks=[make_agent_task()], progress=progress)

        assert isinstance(raw, str)
        data = json.loads(raw)
        assert "results" in data

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_labels_auto_assigned(self, mock_factory, progress):
        """Unlabeled tasks receive unique auto-assigned labels."""
        _setup_mock_runner(mock_factory)

        tasks = [make_agent_task(agent="gemini"), make_agent_task(agent="gemini")]
        raw = await batch_prompt(tasks=tasks, progress=progress)
        data = json.loads(raw)

        labels = [r["label"] for r in data["results"]]
        assert len(set(labels)) == 2  # all unique

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_empty_task_list(self, mock_factory, progress):
        """An empty task list returns total=0 and empty results."""
        raw = await batch_prompt(tasks=[], progress=progress)
        data = json.loads(raw)

        assert data["total"] == 0
        assert data["results"] == []

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_unexpected_exception_captured(self, mock_factory, progress):
        """RuntimeError from runner is captured as task error, not propagated."""
        _setup_mock_runner(mock_factory, side_effect=RuntimeError("unexpected boom"))

        raw = await batch_prompt(tasks=[make_agent_task()], progress=progress)
        data = json.loads(raw)

        assert data["failed"] == 1
        assert "unexpected boom" in data["results"][0]["error"]

    def test_default_concurrency_is_three(self):
        """DEFAULT_MAX_CONCURRENCY constant equals 3."""
        assert DEFAULT_MAX_CONCURRENCY == 3

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_progress_set_total_called_once(self, mock_factory, progress):
        """progress.set_total() is called exactly once with the task count."""
        _setup_mock_runner(mock_factory)

        tasks = [make_agent_task() for _ in range(3)]
        await batch_prompt(tasks=tasks, progress=progress)

        progress.set_total.assert_called_once_with(3)

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_single_task_no_suffix(self, mock_factory, progress):
        """A single task's label is the agent name without any suffix."""
        _setup_mock_runner(mock_factory)

        raw = await batch_prompt(tasks=[make_agent_task(agent="gemini")], progress=progress)
        data = json.loads(raw)

        assert data["results"][0]["label"] == "gemini"
