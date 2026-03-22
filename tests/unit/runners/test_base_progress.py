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
