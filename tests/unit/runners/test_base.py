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
from nexus_mcp.process import run_subprocess
from nexus_mcp.runners.base import AbstractRunner, CLIRunner
from nexus_mcp.types import AgentResponse, PromptRequest
from tests.fixtures import create_mock_process, make_prompt_request


class ConcreteRunner(AbstractRunner):
    """Test implementation of AbstractRunner for testing Template Method pattern."""

    def __init__(self) -> None:
        """Initialize test runner."""
        super().__init__()

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

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_runner_truncates_large_output(self, mock_exec, runner):
        """Runner truncates output exceeding limit and saves to temp file."""
        # Create 100KB output
        large_output = "x" * 100_000
        mock_exec.return_value = create_mock_process(stdout=large_output)

        request = make_prompt_request(prompt="test")

        with patch("nexus_mcp.config.get_global_output_limit", return_value=50_000):
            response = await runner.run(request)

        # Output should be truncated
        assert len(response.output.encode("utf-8")) <= 50_000
        # Metadata should include temp file path
        assert response.metadata.get("truncated") is True
        assert "full_output_path" in response.metadata
        assert response.metadata.get("original_size_bytes") == 100_000

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_runner_preserves_small_output(self, mock_exec, runner):
        """Runner does not truncate output under limit."""
        # Create 1KB output
        small_output = "x" * 1000
        mock_exec.return_value = create_mock_process(stdout=small_output)

        request = make_prompt_request(prompt="test")

        with patch("nexus_mcp.config.get_global_output_limit", return_value=50_000):
            response = await runner.run(request)

        # Output should be preserved
        assert response.output == small_output
        # No truncation metadata
        assert "truncated" not in response.metadata

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_passes_timeout_to_subprocess(self, mock_exec, runner):
        """run() should pass self.timeout to run_subprocess()."""
        mock_exec.return_value = create_mock_process(stdout="output")
        request = make_prompt_request(prompt="test")

        with patch("nexus_mcp.runners.base.run_subprocess", wraps=run_subprocess) as mock_run:
            await runner.run(request)
            mock_run.assert_awaited_once()
            _, kwargs = mock_run.call_args
            assert kwargs.get("timeout") == runner.timeout
