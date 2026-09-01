# tests/unit/test_server_progress.py
"""Tests for compatibility progress forwarding from durable job events."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.core import (
    JobEvent,
    JobResultEnvelope,
    SucceededJobResultResponse,
    TurnResult,
)
from nexus_mcp.emitters import make_progress_emitter
from nexus_mcp.jobs import AgentJobService
from nexus_mcp.server import (
    batch_prompt,
    prompt,
)
from nexus_mcp.types import AgentTask
from tests.fixtures import CODEX_NDJSON_RESPONSE, create_mock_process, make_job_handle


@pytest.fixture(autouse=True)
def _legacy_backend_available(monkeypatch):
    """Keep durable legacy-backend availability independent of host CLI installation."""
    monkeypatch.setattr(
        "nexus_mcp.legacy.runner_backend.detect_cli",
        lambda _backend: CLIInfo(found=True, path="/test/cli"),
    )
    monkeypatch.setattr("nexus_mcp.legacy.runner_backend.get_cli_version", lambda _backend: "test")


class TestMakeProgressEmitter:
    """Verify make_progress_emitter bridges to ctx.report_progress."""

    async def test_emitter_calls_ctx_report_progress(self):
        """Emitter should forward (progress, total, message) to ctx."""
        ctx = AsyncMock(spec=Context)
        emitter = make_progress_emitter(ctx)

        await emitter(2, 5, "Executing subprocess")

        ctx.report_progress.assert_awaited_once_with(
            progress=2, total=5, message="Executing subprocess"
        )

    async def test_emitter_forwards_multiple_calls(self):
        """Multiple calls should each forward to ctx."""
        ctx = AsyncMock(spec=Context)
        emitter = make_progress_emitter(ctx)

        await emitter(1, 5, "Building command")
        await emitter(2, 5, "Executing subprocess")

        assert ctx.report_progress.await_count == 2


class TestMakeBatchProgressEmitter:
    """Verify batch emitter wraps runner progress with task-level counters."""

    async def test_batch_emitter_replaces_progress_total(self):
        """Batch emitter should use task_idx/task_count, not runner's progress/total."""
        ctx = AsyncMock(spec=Context)
        emitter = make_progress_emitter(ctx, task_idx=2, task_count=5, label="summarize")

        await emitter(3, 5, "Parsing output")

        ctx.report_progress.assert_awaited_once_with(
            progress=2,
            total=5,
            message="Task 'summarize' (2/5): Parsing output",
        )

    async def test_batch_emitter_preserves_runner_message(self):
        """Batch emitter should include runner's original message after prefix."""
        ctx = AsyncMock(spec=Context)
        emitter = make_progress_emitter(ctx, task_idx=1, task_count=3, label="analyze")

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
    async def test_single_task_uses_unwrapped_emitter(self, mock_exec, ctx, fast_job_runtime):
        """Single task via batch_prompt should use unwrapped progress emitter."""
        del fast_job_runtime
        mock_exec.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE, returncode=0)
        task = AgentTask(cli="codex", prompt="test", execution_mode="default")

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
    async def test_multi_task_uses_wrapped_emitter(self, mock_exec, ctx, fast_job_runtime):
        """Multi-task batch should use wrapped progress emitter with task prefix."""
        del fast_job_runtime
        mock_exec.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE, returncode=0)
        tasks = [
            AgentTask(cli="codex", prompt="task1", label="first", execution_mode="default"),
            AgentTask(cli="codex", prompt="task2", label="second", execution_mode="default"),
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
    async def test_prompt_reports_progress(self, mock_exec, ctx, fast_job_runtime):
        """prompt() should report runner-level progress (unwrapped)."""
        del fast_job_runtime
        mock_exec.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE, returncode=0)

        await prompt(cli="codex", prompt="test", ctx=ctx)

        # Verify progress was reported
        assert ctx.report_progress.await_count >= 1
        # Should NOT have task wrapper prefix
        first_msg = ctx.report_progress.call_args_list[0].kwargs["message"]
        assert "Task '" not in first_msg


class TestDurableEventBridge:
    """Verify normalized journal events map to FastMCP context exactly once."""

    async def test_forwards_progress_log_and_nonterminal_message(
        self, monkeypatch, ctx, fake_runner_registry
    ):
        async def start(**kwargs):
            return make_job_handle(job_id="job-events", operation=kwargs["operation"])

        async def result(**_kwargs):
            return SucceededJobResultResponse(
                job_id="job-events",
                result=JobResultEnvelope(
                    job_id="job-events", payload=TurnResult(message="terminal output")
                ),
            )

        async def events():
            yield JobEvent(
                job_id="job-events",
                sequence=1,
                type="progress",
                payload={"progress": 2, "total": 5, "message": "Executing"},
            )
            yield JobEvent(
                job_id="job-events",
                sequence=2,
                type="log",
                payload={"level": "warning", "message": "provider warning"},
            )
            yield JobEvent(
                job_id="job-events",
                sequence=3,
                type="message",
                payload={"text": "compatibility notification"},
            )
            yield JobEvent(
                job_id="job-events",
                sequence=4,
                type="message",
                payload={"text": "terminal output", "final": True},
            )
            yield JobEvent(job_id="job-events", sequence=5, type="job_completed")

        start_mock = AsyncMock(side_effect=start)
        result_mock = AsyncMock(side_effect=result)
        subscribe_mock = Mock(return_value=events())
        monkeypatch.setattr(AgentJobService, "start", start_mock)
        monkeypatch.setattr(AgentJobService, "result", result_mock)
        monkeypatch.setattr(AgentJobService, "subscribe_events", subscribe_mock)

        response = await batch_prompt(
            tasks=[AgentTask(cli=fake_runner_registry, prompt="test")], ctx=ctx
        )

        assert response.results[0].output is not None
        ctx.report_progress.assert_awaited_once_with(progress=2.0, total=5.0, message="Executing")
        ctx.warning.assert_awaited_once_with("provider warning")
        info_messages = [call.args[0] for call in ctx.info.await_args_list]
        assert info_messages.count("compatibility notification") == 1
        assert "terminal output" not in info_messages
        subscribe_mock.assert_called_once()
        subscribe_kwargs = subscribe_mock.call_args.kwargs
        assert subscribe_kwargs["job_id"] == "job-events"
        assert subscribe_kwargs["after_sequence"] == 0
