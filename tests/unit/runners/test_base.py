# tests/unit/runners/test_base.py
"""Tests for runner base classes (Protocol + AbstractRunner ABC).

Tests verify:
- CLIRunner Protocol defines required interface
- AbstractRunner implements Template Method pattern
- run() orchestrates: build_command → run_subprocess → parse_output
- run() raises SubprocessError on non-zero return codes (fail fast)
- run() retries on RetryableError with exponential backoff
"""

from unittest.mock import AsyncMock, patch

import pytest

from nexus_mcp.exceptions import ParseError, RetryableError, SubprocessError
from nexus_mcp.process import run_subprocess
from nexus_mcp.runners.base import AbstractRunner, CLIRunner
from nexus_mcp.types import AgentResponse, PromptRequest
from tests.fixtures import create_mock_process, make_agent_response, make_prompt_request


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
    async def test_run_includes_stdout_in_subprocess_error(self, mock_exec, runner):
        """SubprocessError raised on non-zero exit should carry stdout."""
        error_json = '{"error": {"code": 429, "message": "Rate limit exceeded"}}'
        mock_exec.return_value = create_mock_process(
            stdout=error_json,
            stderr="Rate limit exceeded",
            returncode=1,
        )
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.stdout == error_json

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

    def test_make_recovered_response_stamps_metadata(self, runner):
        """_make_recovered_response adds recovery keys to a copy of the response."""
        original = AgentResponse(agent="test", output="ok", raw_output="{}", metadata={"k": 1})

        recovered = runner._make_recovered_response(original, returncode=2, stderr="err text")

        assert recovered.metadata["recovered_from_error"] is True
        assert recovered.metadata["original_exit_code"] == 2
        assert recovered.metadata["stderr"] == "err text"
        # Existing metadata preserved
        assert recovered.metadata["k"] == 1

    def test_make_recovered_response_does_not_mutate_original(self, runner):
        """_make_recovered_response leaves the original response unchanged."""
        original = AgentResponse(agent="test", output="ok", raw_output="{}", metadata={"k": 1})

        runner._make_recovered_response(original, returncode=1, stderr="err")

        assert "recovered_from_error" not in original.metadata

    def test_try_extract_error_default_is_noop(self, runner):
        """Default _try_extract_error returns None and does not raise."""
        result = runner._try_extract_error(
            stdout='{"error": {"code": 429}}',
            stderr="rate limit",
            returncode=1,
            command=["test-cli"],
        )
        assert result is None


class TestRetryLoop:
    """Test AbstractRunner retry loop behavior (run() wraps _execute() with backoff)."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        """Provide test runner instance."""
        return ConcreteRunner()

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_no_retry_on_success(self, mock_exec, runner):
        """Successful execution calls subprocess exactly once."""
        mock_exec.return_value = create_mock_process(stdout="ok output")
        request = make_prompt_request()

        await runner.run(request)

        assert mock_exec.await_count == 1

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retries_on_retryable_error_and_succeeds(self, mock_exec, runner):
        """Retries after RetryableError, returns AgentResponse on success attempt."""
        # Fail twice with RetryableError, succeed on third attempt
        mock_exec.side_effect = [
            create_mock_process(stdout='{"error": {"code": 429}}', returncode=1),
            create_mock_process(stdout='{"error": {"code": 429}}', returncode=1),
            create_mock_process(stdout="success output", returncode=0),
        ]

        # ConcreteRunner has no _try_extract_error override, so returncode=1 raises
        # plain SubprocessError (non-retryable) from _execute. We need a runner that
        # raises RetryableError. Use _execute mock instead.
        success_response = make_agent_response(output="success output")
        retryable = RetryableError("rate limit", returncode=429)

        runner._execute = AsyncMock(  # type: ignore[method-assign]
            side_effect=[retryable, retryable, success_response]
        )
        request = make_prompt_request(max_retries=3)

        response = await runner.run(request)

        assert response.output == "success output"
        assert runner._execute.await_count == 3

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_exhausts_attempts_reraises_retryable_error(self, mock_exec, runner):
        """When all attempts fail with RetryableError, re-raises the last one."""
        retryable = RetryableError("rate limit", returncode=429)
        runner._execute = AsyncMock(side_effect=retryable)  # type: ignore[method-assign]
        request = make_prompt_request(max_retries=3)

        with pytest.raises(RetryableError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 429
        assert runner._execute.await_count == 3

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_no_retry_on_plain_subprocess_error(self, mock_exec, runner):
        """Non-retryable SubprocessError propagates immediately without retry."""
        runner._execute = AsyncMock(  # type: ignore[method-assign]
            side_effect=SubprocessError("auth failed", returncode=401)
        )
        request = make_prompt_request(max_retries=3)

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        # Only one attempt — no retry on SubprocessError
        assert runner._execute.await_count == 1
        assert exc_info.value.returncode == 401

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_no_retry_on_parse_error(self, mock_exec, runner):
        """ParseError propagates immediately without retry."""
        runner._execute = AsyncMock(  # type: ignore[method-assign]
            side_effect=ParseError("bad json")
        )
        request = make_prompt_request(max_retries=3)

        with pytest.raises(ParseError):
            await runner.run(request)

        assert runner._execute.await_count == 1

    def test_compute_backoff_stays_within_max_delay(self, runner):
        """Backoff delay never exceeds max_delay regardless of attempt number."""
        runner.max_delay = 10.0
        runner.base_delay = 2.0

        # At high attempt numbers, cap = min(10, 2 * 2^100) → cap=10
        for attempt in range(20):
            delay = runner._compute_backoff(attempt, retry_after=None)
            assert delay <= runner.max_delay

    def test_compute_backoff_uses_full_jitter(self, runner):
        """Backoff uses random.uniform(0, cap) — result is within [0, cap]."""
        runner.base_delay = 2.0
        runner.max_delay = 60.0

        # attempt=0: cap = min(60, 2 * 2^0) = 2 → delay in [0, 2]
        with patch("nexus_mcp.runners.base.random.uniform", return_value=1.5) as mock_uniform:
            delay = runner._compute_backoff(0, retry_after=None)

        mock_uniform.assert_called_once_with(0, 2.0)
        assert delay == 1.5

    def test_compute_backoff_respects_retry_after_hint(self, runner):
        """retry_after hint is used when larger than computed backoff."""
        runner.base_delay = 2.0
        runner.max_delay = 60.0

        with patch("nexus_mcp.runners.base.random.uniform", return_value=0.5):
            delay = runner._compute_backoff(0, retry_after=30.0)

        # computed=0.5, retry_after=30 → max(0.5, 30) = 30
        assert delay == 30.0

    def test_compute_backoff_computed_wins_when_larger_than_retry_after(self, runner):
        """Computed delay is used when larger than retry_after hint."""
        runner.base_delay = 2.0
        runner.max_delay = 60.0

        with patch("nexus_mcp.runners.base.random.uniform", return_value=1.8):
            delay = runner._compute_backoff(0, retry_after=0.1)

        # computed=1.8, retry_after=0.1 → max(1.8, 0.1) = 1.8
        assert delay == 1.8

    async def test_max_retries_1_means_single_attempt(self, runner):
        """max_retries=1 runs exactly once with no retry on failure."""
        retryable = RetryableError("rate limit", returncode=429)
        runner._execute = AsyncMock(side_effect=retryable)  # type: ignore[method-assign]
        request = make_prompt_request(max_retries=1)

        with pytest.raises(RetryableError):
            await runner.run(request)

        assert runner._execute.await_count == 1

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_max_retries_none_falls_back_to_env_default(self, mock_exec, runner):
        """max_retries=None uses runner.default_max_attempts (from env)."""
        retryable = RetryableError("rate limit")
        runner._execute = AsyncMock(side_effect=retryable)  # type: ignore[method-assign]
        runner.default_max_attempts = 4
        request = make_prompt_request(max_retries=None)

        with pytest.raises(RetryableError):
            await runner.run(request)

        assert runner._execute.await_count == 4

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_backoff_called_between_attempts(self, mock_exec, runner):
        """asyncio.sleep is called (max_attempts - 1) times between attempts."""
        retryable = RetryableError("rate limit")
        runner._execute = AsyncMock(side_effect=retryable)  # type: ignore[method-assign]
        request = make_prompt_request(max_retries=3)

        sleep_calls: list[float] = []

        async def capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("asyncio.sleep", side_effect=capture_sleep), pytest.raises(RetryableError):
            await runner.run(request)

        # 3 attempts → 2 sleeps (no sleep after the last attempt)
        assert len(sleep_calls) == 2
