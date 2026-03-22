# tests/unit/test_server_progress.py
"""Tests for server-side progress emitter factories."""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Context

from nexus_mcp.server import (
    _make_progress_emitter,
    batch_prompt,
    prompt,
)
from nexus_mcp.types import AgentTask
from tests.fixtures import GEMINI_JSON_RESPONSE, create_mock_process


class TestMakeProgressEmitter:
    """Verify _make_progress_emitter bridges to ctx.report_progress."""

    async def test_emitter_calls_ctx_report_progress(self):
        """Emitter should forward (progress, total, message) to ctx."""
        ctx = AsyncMock(spec=Context)
        emitter = _make_progress_emitter(ctx)

        await emitter(2, 5, "Executing subprocess")

        ctx.report_progress.assert_awaited_once_with(
            progress=2, total=5, message="Executing subprocess"
        )

    async def test_emitter_forwards_multiple_calls(self):
        """Multiple calls should each forward to ctx."""
        ctx = AsyncMock(spec=Context)
        emitter = _make_progress_emitter(ctx)

        await emitter(1, 5, "Building command")
        await emitter(2, 5, "Executing subprocess")

        assert ctx.report_progress.await_count == 2


class TestMakeBatchProgressEmitter:
    """Verify batch emitter wraps runner progress with task-level counters."""

    async def test_batch_emitter_replaces_progress_total(self):
        """Batch emitter should use task_idx/task_count, not runner's progress/total."""
        ctx = AsyncMock(spec=Context)
        emitter = _make_progress_emitter(ctx, task_idx=2, task_count=5, label="summarize")

        await emitter(3, 5, "Parsing output")

        ctx.report_progress.assert_awaited_once_with(
            progress=2,
            total=5,
            message="Task 'summarize' (2/5): Parsing output",
        )

    async def test_batch_emitter_preserves_runner_message(self):
        """Batch emitter should include runner's original message after prefix."""
        ctx = AsyncMock(spec=Context)
        emitter = _make_progress_emitter(ctx, task_idx=1, task_count=3, label="analyze")

        await emitter(1, 1, "Attempt 1/1")

        ctx.report_progress.assert_awaited_once_with(
            progress=1,
            total=3,
            message="Task 'analyze' (1/3): Attempt 1/1",
        )


@pytest.mark.usefixtures("mock_cli_detection")
class TestBatchPromptProgressWiring:
    """Verify batch_prompt passes progress emitters to runners."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_single_task_uses_unwrapped_emitter(self, mock_exec, ctx):
        """Single task via batch_prompt should use unwrapped progress emitter."""
        mock_exec.return_value = create_mock_process(stdout=GEMINI_JSON_RESPONSE, returncode=0)
        task = AgentTask(cli="gemini", prompt="test", execution_mode="default")

        await batch_prompt(tasks=[task], ctx=ctx)

        # Verify ctx.report_progress was called with runner-level progress (not task wrapper)
        progress_calls = ctx.report_progress.call_args_list
        assert len(progress_calls) >= 1
        # First runner progress call should have total=5 (step-level), not total=1 (task-level)
        # The attempt-level call comes first: (1, max_attempts, "Attempt 1/N")
        first_call = progress_calls[0]
        assert first_call.kwargs["message"].startswith("Attempt ")
        # Should NOT have task prefix
        assert "Task '" not in first_call.kwargs["message"]

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_multi_task_uses_wrapped_emitter(self, mock_exec, ctx):
        """Multi-task batch should use wrapped progress emitter with task prefix."""
        mock_exec.return_value = create_mock_process(stdout=GEMINI_JSON_RESPONSE, returncode=0)
        tasks = [
            AgentTask(cli="gemini", prompt="task1", label="first", execution_mode="default"),
            AgentTask(cli="gemini", prompt="task2", label="second", execution_mode="default"),
        ]

        await batch_prompt(tasks=tasks, ctx=ctx)

        # Verify progress calls have task-level wrapping
        progress_calls = ctx.report_progress.call_args_list
        messages = [c.kwargs["message"] for c in progress_calls]
        # Should contain wrapped messages from both tasks
        assert any("Task 'first'" in m for m in messages)
        assert any("Task 'second'" in m for m in messages)


@pytest.mark.usefixtures("mock_cli_detection")
class TestPromptProgressWiring:
    """Verify prompt() passes progress through via batch_prompt."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_prompt_reports_progress(self, mock_exec, ctx):
        """prompt() should report runner-level progress (unwrapped)."""
        mock_exec.return_value = create_mock_process(stdout=GEMINI_JSON_RESPONSE, returncode=0)

        await prompt(cli="gemini", prompt="test", ctx=ctx)

        # Verify progress was reported
        assert ctx.report_progress.await_count >= 1
        # Should NOT have task wrapper prefix
        first_msg = ctx.report_progress.call_args_list[0].kwargs["message"]
        assert "Task '" not in first_msg
