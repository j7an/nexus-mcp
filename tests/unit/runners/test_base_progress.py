# tests/unit/runners/test_base_progress.py
"""Tests for ProgressEmitter protocol and runner progress reporting."""

from unittest.mock import AsyncMock, call, patch

import pytest

from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, ProgressEmitter, PromptRequest  # noqa: TC001
from tests.fixtures import create_mock_process, make_prompt_request


class TestProgressEmitterProtocol:
    """Verify ProgressEmitter protocol is satisfied by async callables."""

    def test_async_callable_satisfies_protocol(self):
        """An async callable with (float, float, str) signature satisfies ProgressEmitter."""
        mock: ProgressEmitter = AsyncMock()
        assert callable(mock)

    def test_noop_progress_satisfies_protocol(self):
        """The _noop_progress default satisfies ProgressEmitter."""
        from nexus_mcp.runners.base import _noop_progress

        emitter: ProgressEmitter = _noop_progress
        assert callable(emitter)


class ConcreteRunner(AbstractRunner):
    """Test runner for progress tests."""

    AGENT_NAME = "test"
    _SUPPORTED_MODES = ("default",)

    def build_command(self, request: PromptRequest) -> list[str]:
        return ["test-cli", "-p", request.prompt]

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        from nexus_mcp.exceptions import ParseError

        if not stdout.strip():
            raise ParseError("No output", raw_output=stdout)
        return AgentResponse(cli="test", output=stdout.strip(), raw_output=stdout)


class TestExecuteStepProgress:
    """Verify _execute() reports progress at each template step."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        return ConcreteRunner()

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_happy_path_reports_steps_1_2_4_5(self, mock_exec, runner):
        """Happy path (returncode=0) reports steps 1,2,4,5 — skips step 3."""
        mock_exec.return_value = create_mock_process(stdout="ok", returncode=0)
        request = make_prompt_request()
        progress = AsyncMock()

        await runner._execute(request, AsyncMock(), progress)

        assert progress.await_count == 4
        calls = progress.call_args_list
        assert calls[0] == call(1, 5, "Building command")
        assert calls[1] == call(2, 5, "Executing subprocess")
        assert calls[2] == call(4, 5, "Parsing output")
        assert calls[3] == call(5, 5, "Applying output limits")

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_recovery_path_reports_steps_1_2_3_5(self, mock_exec, runner):
        """Recovery path (non-zero exit, parseable output) reports steps 1,2,3,5."""
        mock_exec.return_value = create_mock_process(stdout="recovered", returncode=1)
        request = make_prompt_request()
        progress = AsyncMock()

        await runner._execute(request, AsyncMock(), progress)

        calls = progress.call_args_list
        assert calls[0] == call(1, 5, "Building command")
        assert calls[1] == call(2, 5, "Executing subprocess")
        assert calls[2] == call(3, 5, "Checking errors")
        assert calls[3] == call(5, 5, "Applying output limits")

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_error_path_reports_steps_1_2_3_then_raises(self, mock_exec, runner):
        """Error path (non-zero exit, unparseable) reports steps 1,2,3 then raises."""
        mock_exec.return_value = create_mock_process(stdout="", returncode=1)
        request = make_prompt_request()
        progress = AsyncMock()

        from nexus_mcp.exceptions import SubprocessError

        with pytest.raises(SubprocessError):
            await runner._execute(request, AsyncMock(), progress)

        calls = progress.call_args_list
        assert calls[0] == call(1, 5, "Building command")
        assert calls[1] == call(2, 5, "Executing subprocess")
        assert calls[2] == call(3, 5, "Checking errors")
        assert len(calls) == 3  # No steps 4-5 on error


from nexus_mcp.exceptions import RetryableError  # noqa: E402


class TestRunRetryProgress:
    """Verify run() reports progress at retry-loop level."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        return ConcreteRunner()

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_single_attempt_reports_attempt_progress(self, mock_exec, runner):
        """Single successful attempt reports attempt 1/1."""
        mock_exec.return_value = create_mock_process(stdout="ok", returncode=0)
        request = make_prompt_request(max_retries=1)
        progress = AsyncMock()

        await runner.run(request, progress=progress)

        # First call should be attempt-level, then step-level calls follow
        assert progress.call_args_list[0] == call(1, 1, "Attempt 1/1")

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_reports_backoff_progress(self, mock_exec, runner):
        """Retry reports backoff wait message between attempts."""
        # First call: RetryableError, second call: success
        mock_exec.side_effect = [
            create_mock_process(stdout="", stderr="rate limited", returncode=1),
            create_mock_process(stdout="ok", returncode=0),
        ]
        # Override _try_extract_error (the intended extension point) to raise
        # RetryableError on the first attempt. _recover_from_error calls this
        # when parse_output fails (empty stdout → ParseError).
        call_count = 0

        def raise_retryable_once(stdout, stderr, returncode, command=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RetryableError("rate limited", stderr=stderr, returncode=returncode)

        runner._try_extract_error = raise_retryable_once

        request = make_prompt_request(max_retries=2)
        progress = AsyncMock()

        await runner.run(request, progress=progress)

        # Check attempt-level progress calls
        call_messages = [c.args[2] for c in progress.call_args_list]
        assert "Attempt 1/2" in call_messages
        assert any("Waiting" in m and "before retry 2/2" in m for m in call_messages)
        assert "Attempt 2/2" in call_messages

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_progress_none_defaults_to_noop(self, mock_exec, runner):
        """run() with progress=None works identically (no-op)."""
        mock_exec.return_value = create_mock_process(stdout="ok", returncode=0)
        request = make_prompt_request()

        # Should not raise — noop progress emitter used
        response = await runner.run(request, progress=None)
        assert response.output == "ok"
