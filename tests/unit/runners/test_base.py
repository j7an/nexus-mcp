# tests/unit/runners/test_base.py
"""Tests for runner base classes (Protocol + AbstractRunner ABC).

Tests verify:
- CLIRunner Protocol defines required interface
- AbstractRunner implements Template Method pattern
- run() orchestrates: build_command → run_subprocess → parse_output
- run() raises SubprocessError on non-zero return codes (fail fast)
"""

from unittest.mock import patch

import pytest

from nexus_mcp.exceptions import SubprocessError
from nexus_mcp.runners.base import AbstractRunner, CLIRunner
from nexus_mcp.types import AgentResponse, PromptRequest
from tests.fixtures import create_mock_process, make_prompt_request


class ConcreteRunner(AbstractRunner):
    """Test implementation of AbstractRunner for testing Template Method pattern."""

    def build_command(self, request: PromptRequest) -> list[str]:
        """Build test command."""
        return ["test-cli", "-p", request.prompt]

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        """Parse test output."""
        return AgentResponse(
            agent="test",
            output=stdout.strip(),
            raw_output=stdout,
        )


class TestCLIRunnerProtocol:
    """Test CLIRunner Protocol defines expected interface."""

    def test_protocol_requires_run_method(self):
        """Protocol should require async run() method."""
        # ConcreteRunner implements CLIRunner Protocol
        runner: CLIRunner = ConcreteRunner()
        assert hasattr(runner, "run")
        assert callable(runner.run)


class TestAbstractRunner:
    """Test AbstractRunner ABC Template Method implementation."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        """Provide test runner instance."""
        return ConcreteRunner()

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_success_orchestrates_template_steps(self, mock_exec, runner):
        """run() should call build_command → run_subprocess → parse_output."""
        # Arrange
        mock_exec.return_value = create_mock_process(
            stdout="success output",
            returncode=0,
        )
        request = make_prompt_request(prompt="test prompt")

        # Act
        response = await runner.run(request)

        # Assert: Template Method orchestration
        mock_exec.assert_awaited_once_with(
            "test-cli",
            "-p",
            "test prompt",
            stdout=-1,  # asyncio.subprocess.PIPE = -1
            stderr=-1,
        )
        assert response.agent == "test"
        assert response.output == "success output"
        assert response.raw_output == "success output"

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_fails_fast_on_nonzero_returncode(self, mock_exec, runner):
        """run() should raise SubprocessError before parse_output on CLI errors."""
        # Arrange: CLI fails with non-zero exit code
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr="CLI error message",
            returncode=1,
        )
        request = make_prompt_request()

        # Act & Assert: Fail fast (parse_output never called)
        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 1
        assert exc_info.value.stderr == "CLI error message"
        assert exc_info.value.command == ["test-cli", "-p", "Hello"]

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_propagates_subprocess_errors(self, mock_exec, runner):
        """run() should propagate SubprocessError from run_subprocess (e.g., FileNotFoundError)."""
        # Arrange: Simulate command not found
        mock_exec.side_effect = FileNotFoundError("test-cli not found")
        request = make_prompt_request()

        # Act & Assert
        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert "Command not found: test-cli" in str(exc_info.value)

    def test_abstract_methods_cannot_be_instantiated(self):
        """AbstractRunner cannot be instantiated directly (requires build_command, parse_output)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AbstractRunner()  # type: ignore[abstract]
