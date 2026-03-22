# tests/unit/test_server_progress.py
"""Tests for server-side progress emitter factories."""

from unittest.mock import AsyncMock

from fastmcp import Context

from nexus_mcp.server import _make_batch_progress_emitter, _make_progress_emitter


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
        emitter = _make_batch_progress_emitter(ctx, task_idx=2, task_count=5, label="summarize")

        await emitter(3, 5, "Parsing output")

        ctx.report_progress.assert_awaited_once_with(
            progress=2,
            total=5,
            message="Task 'summarize' (2/5): Parsing output",
        )

    async def test_batch_emitter_preserves_runner_message(self):
        """Batch emitter should include runner's original message after prefix."""
        ctx = AsyncMock(spec=Context)
        emitter = _make_batch_progress_emitter(ctx, task_idx=1, task_count=3, label="analyze")

        await emitter(1, 1, "Attempt 1/1")

        ctx.report_progress.assert_awaited_once_with(
            progress=1,
            total=3,
            message="Task 'analyze' (1/3): Attempt 1/1",
        )
