# tests/unit/runners/test_base.py
"""Tests for runner base classes (Protocol + AbstractRunner ABC).

Tests verify:
- CLIRunner Protocol defines required interface
- AbstractRunner implements Template Method pattern
- run() orchestrates: build_command → run_subprocess → parse_output
- run() raises SubprocessError on non-zero return codes (fail fast)
- run() retries on RetryableError with exponential backoff
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from nexus_mcp.exceptions import ParseError, RetryableError, SubprocessError
from nexus_mcp.process import run_subprocess
from nexus_mcp.runners.base import AbstractRunner, CLIRunner
from nexus_mcp.types import AgentResponse, PromptRequest
from tests.fixtures import create_mock_process, make_agent_response, make_prompt_request


class ConcreteRunner(AbstractRunner):
    """Test implementation of AbstractRunner for testing Template Method pattern."""

    AGENT_NAME = "test"

    def build_command(self, request: PromptRequest) -> list[str]:
        """Build test command."""
        return ["test-cli", "-p", request.prompt]

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        """Parse test output. Raises ParseError for empty stdout (like real runners)."""
        if not stdout.strip():
            raise ParseError("No output from test-cli", raw_output=stdout)
        return AgentResponse(
            cli="test",
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
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert response.cli == "test"
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
        """Runner truncates output exceeding limit and records size metadata."""
        # Create 100KB output
        large_output = "x" * 100_000
        mock_exec.return_value = create_mock_process(stdout=large_output)

        request = make_prompt_request(prompt="test")

        with patch("nexus_mcp.config.get_global_output_limit", return_value=50_000):
            response = await runner.run(request)

        # Output should be truncated
        assert len(response.output.encode("utf-8")) <= 50_000
        assert response.metadata.get("truncated") is True
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
        """SubprocessError raised on non-zero exit (after failed recovery) should carry stdout.

        Patches parse_output to raise ParseError so recovery fails and the generic
        SubprocessError (raised by _execute) carries the original stdout value.
        """
        error_json = '{"error": {"code": 429, "message": "Rate limit exceeded"}}'
        mock_exec.return_value = create_mock_process(
            stdout=error_json,
            stderr="Rate limit exceeded",
            returncode=1,
        )
        request = make_prompt_request()

        with (
            patch.object(runner, "parse_output", side_effect=ParseError("forced failure")),
            pytest.raises(SubprocessError) as exc_info,
        ):
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
        original = AgentResponse(cli="test", output="ok", raw_output="{}", metadata={"k": 1})

        recovered = runner._make_recovered_response(original, returncode=2, stderr="err text")

        assert recovered.metadata["recovered_from_error"] is True
        assert recovered.metadata["original_exit_code"] == 2
        assert recovered.metadata["stderr"] == "err text"
        # Existing metadata preserved
        assert recovered.metadata["k"] == 1

    def test_make_recovered_response_does_not_mutate_original(self, runner):
        """_make_recovered_response leaves the original response unchanged."""
        original = AgentResponse(cli="test", output="ok", raw_output="{}", metadata={"k": 1})

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

    async def test_retries_on_retryable_error_and_succeeds(self, runner):
        """Retries after RetryableError, returns AgentResponse on success attempt."""
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

    async def test_exhausts_attempts_reraises_retryable_error(self, runner):
        """When all attempts fail with RetryableError, re-raises the last one."""
        retryable = RetryableError("rate limit", returncode=429)
        runner._execute = AsyncMock(side_effect=retryable)  # type: ignore[method-assign]
        request = make_prompt_request(max_retries=3)

        with pytest.raises(RetryableError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 429
        assert runner._execute.await_count == 3

    async def test_no_retry_on_plain_subprocess_error(self, runner):
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

    async def test_no_retry_on_parse_error(self, runner):
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
        with patch("nexus_mcp.runners.retry.random.uniform", return_value=1.5) as mock_uniform:
            delay = runner._compute_backoff(0, retry_after=None)

        mock_uniform.assert_called_once_with(0, 2.0)
        assert delay == 1.5

    def test_compute_backoff_respects_retry_after_hint(self, runner):
        """retry_after hint is used when larger than computed backoff."""
        runner.base_delay = 2.0
        runner.max_delay = 60.0

        with patch("nexus_mcp.runners.retry.random.uniform", return_value=0.5):
            delay = runner._compute_backoff(0, retry_after=30.0)

        # computed=0.5, retry_after=30 → max(0.5, 30) = 30
        assert delay == 30.0

    def test_compute_backoff_computed_wins_when_larger_than_retry_after(self, runner):
        """Computed delay is used when larger than retry_after hint."""
        runner.base_delay = 2.0
        runner.max_delay = 60.0

        with patch("nexus_mcp.runners.retry.random.uniform", return_value=1.8):
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

    async def test_max_retries_none_falls_back_to_env_default(self, runner):
        """max_retries=None uses runner.default_max_attempts (from env)."""
        retryable = RetryableError("rate limit")
        runner._execute = AsyncMock(side_effect=retryable)  # type: ignore[method-assign]
        runner.default_max_attempts = 4
        request = make_prompt_request(max_retries=None)

        with pytest.raises(RetryableError):
            await runner.run(request)

        assert runner._execute.await_count == 4

    async def test_backoff_called_between_attempts(self, runner):
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


class TestBuildPrompt:
    """Test AbstractRunner._build_prompt() with file_refs."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        return ConcreteRunner()

    def test_build_prompt_with_file_refs(self, runner):
        """file_refs appended → output contains file reference section."""
        request = make_prompt_request(prompt="analyze code", file_refs=["a.py", "b.py"])
        result = runner._build_prompt(request)
        assert "File references:\n- a.py\n- b.py" in result
        assert result.startswith("analyze code")

    def test_build_prompt_empty_file_refs(self, runner):
        """file_refs=[] → returns request.prompt unchanged."""
        request = make_prompt_request(prompt="simple prompt", file_refs=[])
        result = runner._build_prompt(request)
        assert result == "simple prompt"


class TestRecoveryMetadata:
    """Test AbstractRunner recovery metadata stamping via run()."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        return ConcreteRunner()

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_recovery_success_through_run(self, mock_exec, runner):
        """returncode=1 + parseable stdout → metadata has recovered_from_error=True,
        original_exit_code=1, stderr=<value>."""
        mock_exec.return_value = create_mock_process(
            stdout="recovered output",
            stderr="some stderr",
            returncode=1,
        )
        request = make_prompt_request(prompt="test")

        response = await runner.run(request)

        assert response.metadata.get("recovered_from_error") is True
        assert response.metadata.get("original_exit_code") == 1
        assert response.metadata.get("stderr") == "some stderr"


class TestTruncationBehavior:
    """Test AbstractRunner output truncation edge cases."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        return ConcreteRunner()

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_truncation_suffix_present(self, mock_exec, runner):
        """100KB output → ends with truncation suffix message."""
        large_output = "x" * 100_000
        mock_exec.return_value = create_mock_process(stdout=large_output)
        request = make_prompt_request(prompt="test")

        with patch("nexus_mcp.config.get_global_output_limit", return_value=50_000):
            response = await runner.run(request)

        expected_suffix = "\n\n[Output truncated: 100000 bytes exceeds 50000 byte limit]"
        assert response.output.endswith(expected_suffix)

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_truncation_at_exact_boundary(self, mock_exec, runner):
        """Output == exactly 50KB → no truncation applied."""
        exact_output = "x" * 50_000
        mock_exec.return_value = create_mock_process(stdout=exact_output)
        request = make_prompt_request(prompt="test")

        with patch("nexus_mcp.config.get_global_output_limit", return_value=50_000):
            response = await runner.run(request)

        assert response.output == exact_output
        assert "truncated" not in response.metadata

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_multibyte_truncation_no_crash(self, mock_exec, runner):
        """Emoji chars near truncation boundary → no UnicodeDecodeError, valid UTF-8 output."""
        # Each emoji is 4 bytes in UTF-8; place them near the 50KB boundary
        # so truncation may split a multi-byte sequence
        emoji = "\U0001f600"  # grinning face, 4 bytes
        # Build ~50KB of emojis, then pad
        emoji_block = emoji * 12_500  # 50_000 bytes exactly
        extra = emoji * 10  # extra content that will be truncated
        full_output = emoji_block + extra
        mock_exec.return_value = create_mock_process(stdout=full_output)
        request = make_prompt_request(prompt="test")

        with patch("nexus_mcp.config.get_global_output_limit", return_value=50_000):
            response = await runner.run(request)

        # Must not raise and result must be valid UTF-8
        response.output.encode("utf-8")


class TestCoerceErrorCode:
    """Test AbstractRunner._coerce_error_code static method."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        return ConcreteRunner()

    def test_int_input_returned_as_is(self, runner):
        """Integer input is returned unchanged."""
        assert runner._coerce_error_code(429) == 429

    def test_string_int_coerced_to_int(self, runner):
        """String that represents an integer is converted to int."""
        assert runner._coerce_error_code("429") == 429

    def test_string_non_int_returned_as_string(self, runner):
        """String that is not a valid integer is returned as-is."""
        result = runner._coerce_error_code("unknown")
        assert result == "unknown"
        assert isinstance(result, str)

    def test_string_float_returned_as_string(self, runner):
        """String that is a float (not int) is returned as-is."""
        result = runner._coerce_error_code("3.14")
        assert result == "3.14"
        assert isinstance(result, str)

    def test_zero_string_coerced_to_zero_int(self, runner):
        """String '0' is converted to integer 0."""
        assert runner._coerce_error_code("0") == 0

    def test_negative_string_int_coerced(self, runner):
        """String '-1' is converted to integer -1."""
        assert runner._coerce_error_code("-1") == -1


class TestRaiseStructuredError:
    """Test AbstractRunner._raise_structured_error method."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        return ConcreteRunner()

    def test_retryable_code_raises_retryable_error(self, runner):
        """Code 429 (in _RETRYABLE_CODES) raises RetryableError."""
        with pytest.raises(RetryableError) as exc_info:
            runner._raise_structured_error(
                "Rate limit exceeded",
                code=429,
                stdout="out",
                stderr="err",
                returncode=429,
                command=["cli"],
            )
        assert "Rate limit exceeded" in str(exc_info.value)

    def test_retryable_code_503_raises_retryable_error(self, runner):
        """Code 503 (in _RETRYABLE_CODES) raises RetryableError."""
        with pytest.raises(RetryableError):
            runner._raise_structured_error(
                "Service unavailable",
                code=503,
                stdout="",
                stderr="",
                returncode=503,
                command=None,
            )

    def test_non_retryable_code_raises_subprocess_error(self, runner):
        """Code 401 (not in _RETRYABLE_CODES) raises SubprocessError."""
        with pytest.raises(SubprocessError) as exc_info:
            runner._raise_structured_error(
                "Unauthorized",
                code=401,
                stdout="out",
                stderr="err",
                returncode=401,
                command=["cli"],
            )
        assert "Unauthorized" in str(exc_info.value)

    def test_string_code_raises_subprocess_error(self, runner):
        """String code (not int) is not in _RETRYABLE_CODES so raises SubprocessError."""
        with pytest.raises(SubprocessError):
            runner._raise_structured_error(
                "API error unknown: oops",
                code="unknown",
                stdout="",
                stderr="",
                returncode=1,
                command=None,
            )

    def test_error_carries_stdout_stderr_returncode(self, runner):
        """Raised error preserves stdout, stderr, and returncode."""
        with pytest.raises(SubprocessError) as exc_info:
            runner._raise_structured_error(
                "Error msg",
                code=400,
                stdout="my stdout",
                stderr="my stderr",
                returncode=400,
                command=["cmd"],
            )
        assert exc_info.value.stdout == "my stdout"
        assert exc_info.value.stderr == "my stderr"
        assert exc_info.value.returncode == 400

    def test_retryable_error_carries_stdout_stderr(self, runner):
        """RetryableError preserves stdout, stderr, and returncode."""
        with pytest.raises(RetryableError) as exc_info:
            runner._raise_structured_error(
                "Rate limit",
                code=429,
                stdout="rate stdout",
                stderr="rate stderr",
                returncode=429,
                command=["cmd"],
            )
        assert exc_info.value.stdout == "rate stdout"
        assert exc_info.value.stderr == "rate stderr"


class TestAbstractRunnerSingleAttempt:
    """Direct tests for the extracted single-attempt helper."""

    @pytest.fixture
    def runner(self) -> ConcreteRunner:
        return ConcreteRunner()

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_execute_single_attempt_recovers_nonzero_exit(self, mock_exec, runner):
        mock_exec.return_value = create_mock_process(
            stdout="success output",
            stderr="minor warning",
            returncode=1,
        )

        response = await runner._execute_single_attempt(
            make_prompt_request(prompt="test"),
            AsyncMock(),
            AsyncMock(),
        )

        assert response.output == "success output"
        assert response.metadata["recovered_from_error"] is True
        assert response.metadata["original_exit_code"] == 1

    async def test_execute_delegates_to_single_attempt(self, runner):
        expected = make_agent_response(output="ok")
        runner._execute_single_attempt = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        response = await runner._execute(make_prompt_request(), AsyncMock(), AsyncMock())

        assert response == expected
        runner._execute_single_attempt.assert_awaited_once()
