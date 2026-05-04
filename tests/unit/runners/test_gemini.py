# tests/unit/runners/test_gemini.py
"""Tests for GeminiRunner.

Tests verify:
- Command building: gemini -p <prompt> --output-format json [--model X] [--yolo]
- Output parsing: {"response": "...", "stats": {...}} → AgentResponse
- Error handling: invalid JSON, missing fields, subprocess errors
"""

import asyncio
from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.exceptions import CLINotFoundError, ParseError, RetryableError, SubprocessError
from nexus_mcp.runners.gemini import GeminiRunner
from tests.fixtures import (
    GEMINI_JSON_RESPONSE,
    GEMINI_JSON_WITH_STATS,
    GEMINI_NOISY_STDOUT,
    create_mock_process,
    gemini_error_json,
    gemini_json,
    make_prompt_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_gemini_runner() -> GeminiRunner:
    """Create a GeminiRunner using the autouse cli_detection_mocks fixture."""
    return GeminiRunner()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestGeminiRunnerInit:
    """Test GeminiRunner CLI detection at construction time."""

    def test_raises_if_cli_not_found(self, mock_cli_detection):
        """GeminiRunner should raise CLINotFoundError if gemini not in PATH."""
        mock_cli_detection.return_value = CLIInfo(found=False)
        with pytest.raises(CLINotFoundError):
            GeminiRunner()

    def test_uses_json_if_supported(self):
        """With version 0.12.0 (autouse fixture), JSON flag should be present."""
        runner = make_gemini_runner()
        cmd = runner.build_command(make_prompt_request(prompt="Test"))
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_omits_json_if_not_supported(self):
        """With old version, JSON flag should be absent."""
        with (
            patch("nexus_mcp.runners.base.detect_cli") as mock_detect,
            patch("nexus_mcp.runners.base.get_cli_version", return_value="0.5.0"),
        ):
            mock_detect.return_value = CLIInfo(found=True, path="/usr/bin/gemini")
            runner = GeminiRunner()
        cmd = runner.build_command(make_prompt_request(prompt="Test"))
        assert "--output-format" not in cmd


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------


class TestGeminiRunnerBuildCommand:
    """Test GeminiRunner.build_command() constructs correct CLI commands."""

    def test_build_command_default_mode(self):
        """Default execution mode should produce: gemini -p <prompt> --output-format json."""
        runner = make_gemini_runner()
        request = make_prompt_request(prompt="Hello, Gemini!")

        command = runner.build_command(request)

        assert command == ["gemini", "-p", "Hello, Gemini!", "--output-format", "json"]

    def test_build_command_with_model(self):
        """Custom model should add --model flag."""
        runner = make_gemini_runner()
        request = make_prompt_request(prompt="test", model="gemini-2.5-pro")

        command = runner.build_command(request)

        assert command == [
            "gemini",
            "-p",
            "test",
            "--output-format",
            "json",
            "--model",
            "gemini-2.5-pro",
        ]

    def test_build_command_all_options(self):
        """All options together: model + yolo mode."""
        runner = make_gemini_runner()
        request = make_prompt_request(
            prompt="complex query",
            model="gemini-2.5-flash",
            execution_mode="yolo",
        )

        command = runner.build_command(request)

        assert command == [
            "gemini",
            "-p",
            "complex query",
            "--output-format",
            "json",
            "--model",
            "gemini-2.5-flash",
            "--yolo",
        ]


# ---------------------------------------------------------------------------
# build_command — execution modes
# ---------------------------------------------------------------------------


class TestGeminiRunnerBuildCommandModes:
    """Test GeminiRunner execution mode flags in build_command()."""

    @pytest.mark.parametrize(
        ("execution_mode", "expected_flag"),
        [
            ("yolo", "--yolo"),
        ],
        ids=["yolo"],
    )
    def test_build_command_execution_mode(self, execution_mode, expected_flag):
        """Execution mode adds corresponding CLI flag."""
        runner = make_gemini_runner()
        request = make_prompt_request(prompt="test", execution_mode=execution_mode)
        command = runner.build_command(request)
        assert command == ["gemini", "-p", "test", "--output-format", "json", expected_flag]


# ---------------------------------------------------------------------------
# File references
# ---------------------------------------------------------------------------


class TestGeminiRunnerFileReferences:
    """Test GeminiRunner file references handling."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_runner_appends_file_refs_to_prompt(self, mock_exec):
        """GeminiRunner appends file references to prompt text."""
        mock_exec.return_value = create_mock_process(stdout='{"response": "ok"}')

        runner = make_gemini_runner()
        request = make_prompt_request(
            prompt="Analyze this code", file_refs=["src/main.py", "tests/test_main.py"]
        )
        await runner.run(request)

        # Verify command includes modified prompt
        args = mock_exec.call_args[0]
        prompt_arg = args[2]  # args = ("gemini", "-p", "<prompt>", ...)
        assert "src/main.py" in prompt_arg
        assert "tests/test_main.py" in prompt_arg
        assert "Analyze this code" in prompt_arg

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_runner_prompt_unchanged_when_no_file_refs(self, mock_exec):
        """GeminiRunner does not modify prompt if file_refs is empty."""
        mock_exec.return_value = create_mock_process(stdout='{"response": "ok"}')

        runner = make_gemini_runner()
        request = make_prompt_request(prompt="Simple prompt")
        await runner.run(request)

        args = mock_exec.call_args[0]
        prompt_arg = args[2]
        assert prompt_arg == "Simple prompt"  # Unchanged


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------


class TestGeminiRunnerParseOutput:
    """Test GeminiRunner.parse_output() handles Gemini CLI JSON responses."""

    def test_parse_output_minimal_json(self):
        """Minimal valid JSON: {"response": "text"} → AgentResponse."""
        runner = make_gemini_runner()

        response = runner.parse_output(GEMINI_JSON_RESPONSE, stderr="")

        assert response.cli == "gemini"
        assert response.output == "test output"
        assert response.raw_output == GEMINI_JSON_RESPONSE
        assert response.metadata == {}

    def test_parse_output_with_stats(self):
        """JSON with stats: {"response": "...", "stats": {...}} → metadata."""
        runner = make_gemini_runner()

        response = runner.parse_output(GEMINI_JSON_WITH_STATS, stderr="")

        assert response.cli == "gemini"
        assert response.output == "Hello, world!"
        assert response.raw_output == GEMINI_JSON_WITH_STATS
        assert response.metadata == {"models": {"gemini-2.5-flash": 1}}

    def test_parse_output_strips_whitespace(self):
        """Response text should be stripped of leading/trailing whitespace."""
        runner = make_gemini_runner()
        json_with_whitespace = '{"response": "  trimmed  "}'

        response = runner.parse_output(json_with_whitespace, stderr="")

        assert response.output == "trimmed"

    def test_parse_output_preserves_internal_whitespace(self):
        """Internal whitespace and newlines should be preserved."""
        runner = make_gemini_runner()
        json_with_newlines = '{"response": "Line 1\\n\\nLine 2"}'

        response = runner.parse_output(json_with_newlines, stderr="")

        assert response.output == "Line 1\n\nLine 2"


# ---------------------------------------------------------------------------
# parse_output — edge cases
# ---------------------------------------------------------------------------


class TestGeminiRunnerParseOutputEdgeCases:
    """Test GeminiRunner.parse_output() edge cases with unusual stats values.

    NOTE: These tests document a known contract deviation — parse_output
    leaks Pydantic ValidationError instead of raising ParseError for
    invalid stats types (stats=None, stats=<non-dict>).
    """

    @pytest.fixture
    def runner(self) -> GeminiRunner:
        return make_gemini_runner()

    def test_parse_output_null_stats_returns_empty_metadata(self, runner):
        """'{"response":"text","stats":null}' → success with metadata={}.

        data.get("stats") returns None (the actual value); `or {}` coerces
        falsy None to empty dict, so AgentResponse is created successfully.
        """
        import json

        stdout = json.dumps({"response": "text", "stats": None})
        result = runner.parse_output(stdout, "")
        assert result.output == "text"
        assert result.metadata == {}

    def test_parse_output_non_dict_stats_raises_validation_error(self, runner):
        """'{"response":"text","stats":42}' → Pydantic ValidationError (stats must be dict)."""
        import json

        from pydantic import ValidationError

        stdout = json.dumps({"response": "text", "stats": 42})
        with pytest.raises(ValidationError):
            runner.parse_output(stdout, "")


# ---------------------------------------------------------------------------
# Scalar JSON (null, number)
# ---------------------------------------------------------------------------


class TestGeminiRunnerScalarJson:
    """Test parse_output handles scalar JSON values (null, number) without TypeError."""

    def test_parse_output_null_json_raises_parse_error(self):
        """json.loads('null') returns None; 'response' not in None raises TypeError without fix."""
        runner = make_gemini_runner()

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output("null", stderr="")

        assert "Missing 'response' field" in str(exc_info.value)

    def test_parse_output_number_json_raises_parse_error(self):
        """json.loads('42') returns int; 'response' not in 42 raises TypeError without fix."""
        runner = make_gemini_runner()

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output("42", stderr="")

        assert "Missing 'response' field" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Noisy stdout
# ---------------------------------------------------------------------------


class TestGeminiRunnerNoisyStdout:
    """Test GeminiRunner.parse_output() handles Node.js warning prefix before JSON."""

    def test_parse_output_with_node_warnings(self):
        """Node.js deprecation warnings before valid JSON → should parse successfully."""
        runner = make_gemini_runner()

        response = runner.parse_output(GEMINI_NOISY_STDOUT, stderr="")

        assert response.cli == "gemini"
        assert response.output == "test output"
        assert response.raw_output == GEMINI_NOISY_STDOUT

    def test_parse_output_with_multiple_prefix_lines(self):
        """Multiple non-JSON lines before JSON → should extract and parse correctly."""
        runner = make_gemini_runner()
        noisy = (
            "info: Starting up\n"
            "debug: Connecting to API\n"
            "warn: Retrying request\n"
            '{"response": "result from noisy stdout"}'
        )

        response = runner.parse_output(noisy, stderr="")

        assert response.output == "result from noisy stdout"

    def test_parse_output_with_prefix_and_stats(self):
        """Prefix noise + JSON with stats → metadata preserved after extraction."""
        runner = make_gemini_runner()
        noisy = (
            "(node:12345) [DEP0040] DeprecationWarning: punycode is deprecated\n"
            "Loaded cached credentials.\n"
            '{"response": "Hello!", "stats": {"models": {"gemini-2.5-flash": 1}}}'
        )

        response = runner.parse_output(noisy, stderr="")

        assert response.output == "Hello!"
        assert response.metadata == {"models": {"gemini-2.5-flash": 1}}

    def test_parse_output_noisy_no_json_raises(self):
        """Prefix noise with NO valid JSON → ParseError (not a regression)."""
        runner = make_gemini_runner()
        noisy_no_json = (
            "(node:87799) [DEP0040] DeprecationWarning: punycode is deprecated\n"
            "Loaded cached credentials.\n"
            "Error: Something went wrong without JSON output"
        )

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(noisy_no_json, stderr="")

        assert exc_info.value.raw_output == noisy_no_json


# ---------------------------------------------------------------------------
# Error handling (_recover_from_error, _try_extract_error unit tests)
# ---------------------------------------------------------------------------


class TestGeminiRunnerErrorHandling:
    """Test GeminiRunner error handling for invalid/malformed output."""

    def test_parse_output_invalid_json(self):
        """Invalid JSON should raise ParseError with context."""
        runner = make_gemini_runner()
        invalid_json = "not valid json {{"

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(invalid_json, stderr="")

        assert "Invalid JSON" in str(exc_info.value)
        assert exc_info.value.raw_output == invalid_json

    def test_parse_output_missing_response_field(self):
        """Missing 'response' field should raise ParseError."""
        runner = make_gemini_runner()
        missing_field = '{"stats": {}}'  # No "response" key

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(missing_field, stderr="")

        assert "Missing 'response' field" in str(exc_info.value)
        assert exc_info.value.raw_output == missing_field

    def test_parse_output_non_string_response(self):
        """Non-string 'response' value should raise ParseError."""
        runner = make_gemini_runner()
        non_string_response = '{"response": 123}'  # response should be string

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(non_string_response, stderr="")

        assert "'response' field must be a string" in str(exc_info.value)
        assert exc_info.value.raw_output == non_string_response


# ---------------------------------------------------------------------------
# Retryable errors
# ---------------------------------------------------------------------------


class TestGeminiRunnerRetryableErrors:
    """Test GeminiRunner raises RetryableError for transient error codes (429, 503).

    These tests verify _try_extract_error classifies codes correctly and
    that the retry loop in run() triggers on RetryableError.
    """

    @pytest.mark.parametrize(
        ("code", "message", "status"),
        [
            (429, "Quota exceeded", "RESOURCE_EXHAUSTED"),
            (503, "Service unavailable", "UNAVAILABLE"),
        ],
        ids=["429-rate-limit", "503-unavailable"],
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retryable_code_raises_retryable_error(self, mock_exec, code, message, status):
        """Retryable HTTP codes raise RetryableError, not plain SubprocessError."""
        mock_exec.return_value = create_mock_process(
            stdout=gemini_error_json(code, message, status), stderr="", returncode=1
        )
        runner = make_gemini_runner()
        request = make_prompt_request(max_retries=1)

        with pytest.raises(RetryableError) as exc_info:
            await runner.run(request)

        assert str(code) in exc_info.value.args[0]

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_401_raises_non_retryable_subprocess_error(self, mock_exec):
        """HTTP 401 (auth failure) → plain SubprocessError, NOT RetryableError."""
        mock_exec.return_value = create_mock_process(
            stdout=gemini_error_json(401, "API key not valid", "UNAUTHENTICATED"),
            stderr="",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request(max_retries=1)

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        # Verify it's NOT a RetryableError subclass
        assert not isinstance(exc_info.value, RetryableError)
        assert "401" in exc_info.value.args[0]

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gaxios_429_in_stderr_raises_retryable_error(self, mock_exec):
        """GaxiosError 429 embedded in stderr JSON array → RetryableError."""
        stderr_with_429 = (
            "Gemini CLI encountered an API error\n"
            f"[{gemini_error_json(429, 'No capacity', 'RESOURCE_EXHAUSTED')}]"
        )
        mock_exec.return_value = create_mock_process(
            stdout="", stderr=stderr_with_429, returncode=1
        )
        runner = make_gemini_runner()
        request = make_prompt_request(max_retries=1)

        with pytest.raises(RetryableError) as exc_info:
            await runner.run(request)

        assert "429" in exc_info.value.args[0]

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_on_429_succeeds_on_second_attempt(self, mock_exec):
        """Full integration: first call returns 429 RetryableError, second succeeds."""
        mock_exec.side_effect = [
            create_mock_process(
                stdout=gemini_error_json(429, "Quota exceeded", "RESOURCE_EXHAUSTED"),
                stderr="",
                returncode=1,
            ),
            create_mock_process(stdout='{"response": "Success after retry"}', returncode=0),
        ]
        runner = make_gemini_runner()
        request = make_prompt_request(max_retries=2)

        response = await runner.run(request)

        assert response.output == "Success after retry"
        assert mock_exec.await_count == 2

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_exhausted_all_429(self, mock_exec):
        """3x 429 errors, max_retries=3 → RetryableError raised, called 3 times."""
        error_stdout = gemini_error_json(429, "Quota exhausted", "RESOURCE_EXHAUSTED")
        mock_exec.return_value = create_mock_process(stdout=error_stdout, returncode=1)
        runner = make_gemini_runner()
        request = make_prompt_request(max_retries=3)

        with pytest.raises(RetryableError):
            await runner.run(request)

        assert mock_exec.await_count == 3


# ---------------------------------------------------------------------------
# Error recovery (full run() with mocked subprocess)
# ---------------------------------------------------------------------------


class TestGeminiRunnerErrorRecovery:
    """Test GeminiRunner error recovery from non-zero exit codes."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_runner_recovers_from_partial_error(self, mock_exec):
        """GeminiRunner recovers when CLI exits with error but stdout has valid JSON."""
        # Scenario: Gemini CLI encountered an issue but still produced partial response
        mock_exec.return_value = create_mock_process(
            stdout='{"response": "Partial result before error", "stats": {}}',
            stderr="Warning: Rate limit approaching",
            returncode=1,
        )

        runner = make_gemini_runner()
        request = make_prompt_request(prompt="test")

        # Should NOT raise - recovers from error by parsing stdout
        response = await runner.run(request)
        assert response.output == "Partial result before error"
        assert response.metadata.get("recovered_from_error") is True
        assert response.metadata.get("original_exit_code") == 1
        assert response.metadata.get("stderr") == "Warning: Rate limit approaching"

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_runner_raises_when_recovery_impossible(self, mock_exec):
        """GeminiRunner raises when stdout is not parseable."""
        mock_exec.return_value = create_mock_process(
            stdout="Fatal error: API key invalid",  # Not JSON
            stderr="Error: 401 Unauthorized",
            returncode=1,
        )

        runner = make_gemini_runner()
        request = make_prompt_request(prompt="test")

        # Should raise - no recovery possible
        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 1
        assert "Error: 401 Unauthorized" in exc_info.value.stderr


# ---------------------------------------------------------------------------
# API error extraction
# ---------------------------------------------------------------------------


class TestGeminiRunnerAPIErrorExtraction:
    """Test GeminiRunner extracts structured API error details from error JSON."""

    @pytest.mark.parametrize(
        ("code", "message", "status"),
        [
            (429, "Quota exceeded for quota metric", "RESOURCE_EXHAUSTED"),
            (401, "API key not valid", "UNAUTHENTICATED"),
        ],
        ids=["429-rate-limit", "401-auth"],
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_error_json_surfaces_in_subprocess_error(
        self, mock_exec, code, message, status
    ):
        """API error JSON produces a structured SubprocessError with code and status."""
        mock_exec.return_value = create_mock_process(
            stdout=gemini_error_json(code, message, status),
            stderr="",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert str(code) in primary_message
        assert status in primary_message
        assert "Gemini API error" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_error_json_preserves_returncode(self, mock_exec):
        """SubprocessError from error JSON should preserve returncode and stdout."""
        error_stdout = gemini_error_json(429, "Rate limited", "RESOURCE_EXHAUSTED")
        mock_exec.return_value = create_mock_process(
            stdout=error_stdout,
            stderr="rate limit hit",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 1
        assert exc_info.value.stdout == error_stdout

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_stderr_json_fallback_when_stdout_empty(self, mock_exec):
        """When stdout is empty but stderr has JSON error block, extracts structured error."""
        stderr_with_json = "Gemini CLI error log\nStack trace...\n" + gemini_error_json(
            429, "Quota exceeded", "RESOURCE_EXHAUSTED"
        )
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr=stderr_with_json,
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "429" in primary_message
        assert "RESOURCE_EXHAUSTED" in primary_message
        assert "Gemini API error" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_stdout_takes_priority_over_stderr_json(self, mock_exec):
        """When both stdout and stderr have JSON error, stdout wins (richer API format)."""
        stdout_error = gemini_error_json(429, "Quota exceeded for stdout", "RESOURCE_EXHAUSTED")
        stderr_with_json = "log line\n" + gemini_error_json(1, "Generic exit code error", "UNKNOWN")
        mock_exec.return_value = create_mock_process(
            stdout=stdout_error,
            stderr=stderr_with_json,
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "429" in primary_message
        assert "Quota exceeded for stdout" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_no_json_anywhere_falls_through_to_generic(self, mock_exec):
        """When neither stdout nor stderr has JSON, falls through to generic SubprocessError."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr="Connection refused: unable to reach API",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "Gemini API error" not in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_malformed_stderr_json_falls_through(self, mock_exec):
        """When stderr has truncated/malformed JSON, falls through gracefully without crash."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr='Some error\n{"error": {"code": 429, "message": "truncated...',
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "Gemini API error" not in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_non_dict_json_stdout_does_not_crash(self, mock_exec):
        """Non-dict JSON in stdout (e.g. array) must not raise AttributeError.

        Previously, _try_extract_error called json.loads(stdout) and then data.get("error")
        without checking isinstance(data, dict), crashing on list/number JSON.
        """
        mock_exec.return_value = create_mock_process(
            stdout="[1, 2, 3]",
            stderr="something went wrong",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        # Should fall through to generic error, NOT crash with AttributeError
        primary_message = exc_info.value.args[0]
        assert "Gemini API error" not in primary_message
        assert "CLI command failed" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_structured_error_includes_command(self, mock_exec):
        """SubprocessError raised from structured API error JSON should include the command."""
        mock_exec.return_value = create_mock_process(
            stdout=gemini_error_json(429, "Rate limited", "RESOURCE_EXHAUSTED"),
            stderr="",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        # The command field should be populated (not None), matching generic error behavior
        assert exc_info.value.command is not None
        assert "gemini" in exc_info.value.command[0]


# ---------------------------------------------------------------------------
# Dual-field recovery
# ---------------------------------------------------------------------------


class TestGeminiRunnerDualFieldRecovery:
    """Test _recover_from_error behaviour when stdout has both 'response' and 'error'."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_response_field_wins_over_error_field_on_nonzero_exit(self, mock_exec):
        """When stdout has both 'response' and 'error', recovery path returns the response.

        Documents the 'recovery wins' contract: parse_output runs before _try_extract_error
        in _recover_from_error, so a valid 'response' field takes priority.
        """
        dual_field_stdout = (
            '{"response": "Partial answer", "error": {"code": 429, "message": "quota"}}'
        )
        mock_exec.return_value = create_mock_process(
            stdout=dual_field_stdout,
            stderr="",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        response = await runner.run(request)

        assert response.output == "Partial answer"
        assert response.metadata.get("recovered_from_error") is True


# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------


class TestGeminiRunnerEnvConfiguration:
    """Test GeminiRunner environment variable configuration."""

    @patch.dict("os.environ", {"NEXUS_GEMINI_MODEL": "gemini-2.5-flash"})
    def test_gemini_runner_uses_default_model_from_env(self):
        """GeminiRunner uses NEXUS_GEMINI_MODEL as default if request.model is None."""
        runner = make_gemini_runner()
        request = make_prompt_request(prompt="test", model=None)

        cmd = runner.build_command(request)
        assert "--model" in cmd
        assert "gemini-2.5-flash" in cmd

    @patch.dict("os.environ", {"NEXUS_GEMINI_MODEL": "gemini-2.5-flash"})
    def test_gemini_runner_request_model_overrides_env(self):
        """Request model overrides env var default."""
        runner = make_gemini_runner()
        request = make_prompt_request(prompt="test", model="gemini-3-pro-preview")

        cmd = runner.build_command(request)
        assert "gemini-3-pro-preview" in cmd
        assert "gemini-2.5-flash" not in cmd


# ---------------------------------------------------------------------------
# Integration (full run() with mocked subprocess)
# ---------------------------------------------------------------------------


class TestGeminiRunnerIntegration:
    """Test GeminiRunner.run() end-to-end with mocked subprocess."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_success_returns_parsed_response(self, mock_exec):
        """Successful run should execute CLI and return parsed AgentResponse."""
        # Arrange
        mock_exec.return_value = create_mock_process(
            stdout=GEMINI_JSON_WITH_STATS,
            returncode=0,
        )
        runner = make_gemini_runner()
        request = make_prompt_request(
            cli="gemini",
            prompt="test prompt",
            model="gemini-2.5-flash",
        )

        # Act
        response = await runner.run(request)

        # Assert: Command built correctly
        mock_exec.assert_awaited_once_with(
            "gemini",
            "-p",
            "test prompt",
            "--output-format",
            "json",
            "--model",
            "gemini-2.5-flash",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Assert: Response parsed correctly
        assert response.cli == "gemini"
        assert response.output == "Hello, world!"
        assert response.metadata["models"]["gemini-2.5-flash"] == 1

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_subprocess_error_propagates(self, mock_exec):
        """Subprocess errors (non-zero exit) should raise SubprocessError."""
        # Arrange: CLI exits with error
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr="API key not found",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        # Act & Assert
        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 1
        assert "API key not found" in exc_info.value.stderr


# ---------------------------------------------------------------------------
# GaxiosError extraction (Gemini-specific)
# ---------------------------------------------------------------------------

# Realistic GaxiosError stderr emitted by Gemini CLI on HTTP 429.
# The real error arrives in a JSON array; the session summary (buggy garbage)
# follows as a bare JSON object appended by the CLI after the array.
_GAXIOS_STDERR_429_ONLY = (
    "Gemini CLI encountered an API error\n"
    '[{"error": {"code": 429, "message": "No capacity available for gemini-2.5-pro",'
    ' "status": "RESOURCE_EXHAUSTED"}}]'
)
_SESSION_SUMMARY_ONLY = '{"error": {"code": 1, "message": "[object Object]", "status": ""}}'
_GAXIOS_PLUS_SUMMARY = (
    "Gemini CLI encountered an API error\n"
    '[{"error": {"code": 429, "message": "No capacity available for gemini-2.5-pro",'
    ' "status": "RESOURCE_EXHAUSTED"}}]\n'
    '{"error": {"code": 1, "message": "[object Object]", "status": ""}}'
)


class TestGaxiosErrorExtraction:
    """Test _try_extract_error surfaces real GaxiosError from stderr JSON array.

    Covers the fix for Issue #14: GaxiosError 429 embedded in a JSON array
    was being ignored in favour of the session summary bare object that follows it.
    """

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gaxios_array_in_stderr_raises_429(self, mock_exec):
        """GaxiosError array in stderr (no session summary) → SubprocessError with '429'."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr=_GAXIOS_STDERR_429_ONLY,
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "429" in primary_message
        assert "Gemini API error" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_session_summary_only_falls_through_to_generic(self, mock_exec):
        """Session summary alone (code=1, message='[object Object]') → generic SubprocessError.

        The guard `if code == 1 and message == "[object Object]": return` must reject
        the known-buggy session summary so it does not produce a misleading message.
        """
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr=_SESSION_SUMMARY_ONLY,
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "Gemini API error" not in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gaxios_array_wins_over_session_summary(self, mock_exec):
        """When both GaxiosError array and session summary are in stderr, array wins."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr=_GAXIOS_PLUS_SUMMARY,
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "429" in primary_message
        assert "Gemini API error" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_real_error_message_preserved(self, mock_exec):
        """The full error message text from the GaxiosError array is preserved in the exception."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr=_GAXIOS_PLUS_SUMMARY,
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "No capacity available for gemini-2.5-pro" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_scalar_error_field_falls_through_to_generic_error(self, mock_exec):
        """When 'error' is a scalar (string) rather than a dict, _try_extract_error returns early.

        Line 204 guards: `if not isinstance(error, dict): return`. A scalar error field
        (e.g. {"error": "something went wrong"}) should not raise a structured SubprocessError
        and instead fall through to the generic CLI command failed message.
        """
        mock_exec.return_value = create_mock_process(
            stdout='{"error": "something went wrong"}',
            stderr="",
            returncode=1,
        )
        runner = make_gemini_runner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "Gemini API error" not in primary_message
        assert "CLI command failed" in primary_message


class TestGeminiRunnerClassConstants:
    def test_supported_modes_class_constant(self):
        assert GeminiRunner._SUPPORTED_MODES == ("default", "yolo")


# ---------------------------------------------------------------------------
# Fallback chain — success path
# ---------------------------------------------------------------------------


class TestGeminiRunnerFallbackSuccess:
    """Retryable primary failure should switch models immediately."""

    @patch.dict(
        "os.environ",
        {"NEXUS_GEMINI_FALLBACK_MODELS": ("gemini-3.1-pro-preview,gemini-3-flash-preview")},
        clear=False,
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retryable_primary_failure_switches_to_fallback_model(self, mock_exec):
        mock_exec.side_effect = [
            create_mock_process(
                stdout=gemini_error_json(429, "No capacity", "RESOURCE_EXHAUSTED"),
                returncode=1,
            ),
            create_mock_process(
                stdout=gemini_json(
                    "fallback output",
                    {"models": {"gemini-3-flash-preview": 1}},
                ),
                returncode=0,
            ),
        ]
        calls: list[tuple[str, str]] = []

        async def collecting_emitter(level: str, message: str) -> None:
            calls.append((level, message))

        runner = make_gemini_runner()
        response = await runner.run(
            make_prompt_request(model="gemini-3.1-pro-preview", max_retries=1),
            emitter=collecting_emitter,
        )

        models = [call.args[call.args.index("--model") + 1] for call in mock_exec.call_args_list]
        assert models == ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]
        assert response.metadata["effective_model"] == "gemini-3-flash-preview"
        assert response.metadata["fallback_model_used"] is True
        assert response.metadata["original_model"] == "gemini-3.1-pro-preview"
        assert response.metadata["fallback_attempt"] == 2
        assert response.metadata["fallback_reason"] == "RESOURCE_EXHAUSTED"
        assert any(
            "switching to fallback model gemini-3-flash-preview" in msg
            for level, msg in calls
            if level == "warning"
        )

    @patch.dict(
        "os.environ",
        {
            "NEXUS_GEMINI_FALLBACK_MODELS": (
                "gemini-3.1-pro-preview,gemini-3-flash-preview,gemini-3-flash-preview"
            )
        },
        clear=False,
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_fallback_chain_deduplicates_while_preserving_order(self, mock_exec):
        mock_exec.side_effect = [
            create_mock_process(
                stdout=gemini_error_json(429, "No capacity", "RESOURCE_EXHAUSTED"),
                returncode=1,
            ),
            create_mock_process(stdout=gemini_json("ok from fallback"), returncode=0),
        ]

        runner = make_gemini_runner()
        await runner.run(make_prompt_request(model="gemini-3.1-pro-preview", max_retries=1))

        models = [call.args[call.args.index("--model") + 1] for call in mock_exec.call_args_list]
        assert models == ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]


# ---------------------------------------------------------------------------
# Fallback chain — retry semantics
# ---------------------------------------------------------------------------


class TestGeminiRunnerFallbackRetrySemantics:
    """Fallback should only advance on RetryableError."""

    @patch.dict(
        "os.environ",
        {"NEXUS_GEMINI_FALLBACK_MODELS": "gemini-3-flash-preview"},
        clear=False,
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_full_chain_retryable_exhaustion_retries_whole_chain(self, mock_exec):
        error_stdout = gemini_error_json(429, "Quota exhausted", "RESOURCE_EXHAUSTED")
        mock_exec.return_value = create_mock_process(stdout=error_stdout, returncode=1)

        runner = make_gemini_runner()
        request = make_prompt_request(model="gemini-3.1-pro-preview", max_retries=2)

        with pytest.raises(RetryableError):
            await runner.run(request)

        models = [call.args[call.args.index("--model") + 1] for call in mock_exec.call_args_list]
        assert models == [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
        ]

    @patch.dict(
        "os.environ",
        {"NEXUS_GEMINI_FALLBACK_MODELS": "gemini-3-flash-preview"},
        clear=False,
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_non_retryable_error_does_not_advance_to_fallback(self, mock_exec):
        mock_exec.return_value = create_mock_process(
            stdout=gemini_error_json(401, "Bad API key", "UNAUTHENTICATED"),
            returncode=1,
        )

        runner = make_gemini_runner()

        with pytest.raises(SubprocessError):
            await runner.run(make_prompt_request(model="gemini-3.1-pro-preview", max_retries=2))

        assert mock_exec.await_count == 1

    @patch.dict(
        "os.environ",
        {"NEXUS_GEMINI_FALLBACK_MODELS": "gemini-3-flash-preview"},
        clear=False,
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_no_primary_model_keeps_existing_retry_only_behavior(self, mock_exec):
        mock_exec.side_effect = [
            create_mock_process(
                stdout=gemini_error_json(429, "Rate limited", "RESOURCE_EXHAUSTED"),
                returncode=1,
            ),
            create_mock_process(stdout=gemini_json("ok after retry"), returncode=0),
        ]

        runner = make_gemini_runner()
        response = await runner.run(make_prompt_request(model=None, max_retries=2))

        assert response.output == "ok after retry"
        assert response.metadata == {}
        for call in mock_exec.call_args_list:
            assert "--model" not in list(call.args)
