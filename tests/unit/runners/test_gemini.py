# tests/unit/runners/test_gemini.py
"""Tests for GeminiRunner.

Tests verify:
- Command building: gemini -p <prompt> --output-format json [--model X] [--sandbox/--yolo]
- Output parsing: {"response": "...", "stats": {...}} → AgentResponse
- Error handling: invalid JSON, missing fields, subprocess errors
"""

from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.exceptions import CLINotFoundError, ParseError, SubprocessError
from nexus_mcp.runners.gemini import GeminiRunner
from tests.fixtures import (
    GEMINI_JSON_RESPONSE,
    GEMINI_JSON_WITH_STATS,
    GEMINI_NOISY_STDOUT,
    create_mock_process,
    make_prompt_request,
)


class TestGeminiRunnerCommandBuilding:
    """Test GeminiRunner.build_command() constructs correct CLI commands."""

    def test_build_command_default_mode(self):
        """Default execution mode should produce: gemini -p <prompt> --output-format json."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="Hello, Gemini!")

        command = runner.build_command(request)

        assert command == ["gemini", "-p", "Hello, Gemini!", "--output-format", "json"]

    def test_build_command_with_model(self):
        """Custom model should add --model flag."""
        runner = GeminiRunner()
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

    def test_build_command_sandbox_mode(self):
        """Sandbox mode should add --sandbox flag."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="test", execution_mode="sandbox")

        command = runner.build_command(request)

        assert command == [
            "gemini",
            "-p",
            "test",
            "--output-format",
            "json",
            "--sandbox",
        ]

    def test_build_command_yolo_mode(self):
        """YOLO mode should add --yolo flag."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="test", execution_mode="yolo")

        command = runner.build_command(request)

        assert command == [
            "gemini",
            "-p",
            "test",
            "--output-format",
            "json",
            "--yolo",
        ]

    def test_build_command_all_options(self):
        """All options together: model + yolo mode."""
        runner = GeminiRunner()
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


class TestGeminiRunnerOutputParsing:
    """Test GeminiRunner.parse_output() handles Gemini CLI JSON responses."""

    def test_parse_output_minimal_json(self):
        """Minimal valid JSON: {"response": "text"} → AgentResponse."""
        runner = GeminiRunner()

        response = runner.parse_output(GEMINI_JSON_RESPONSE, stderr="")

        assert response.agent == "gemini"
        assert response.output == "test output"
        assert response.raw_output == GEMINI_JSON_RESPONSE
        assert response.metadata == {}

    def test_parse_output_with_stats(self):
        """JSON with stats: {"response": "...", "stats": {...}} → metadata."""
        runner = GeminiRunner()

        response = runner.parse_output(GEMINI_JSON_WITH_STATS, stderr="")

        assert response.agent == "gemini"
        assert response.output == "Hello, world!"
        assert response.raw_output == GEMINI_JSON_WITH_STATS
        assert response.metadata == {"models": {"gemini-2.5-flash": 1}}

    def test_parse_output_strips_whitespace(self):
        """Response text should be stripped of leading/trailing whitespace."""
        runner = GeminiRunner()
        json_with_whitespace = '{"response": "  trimmed  "}'

        response = runner.parse_output(json_with_whitespace, stderr="")

        assert response.output == "trimmed"

    def test_parse_output_preserves_internal_whitespace(self):
        """Internal whitespace and newlines should be preserved."""
        runner = GeminiRunner()
        json_with_newlines = '{"response": "Line 1\\n\\nLine 2"}'

        response = runner.parse_output(json_with_newlines, stderr="")

        assert response.output == "Line 1\n\nLine 2"


class TestGeminiRunnerErrorHandling:
    """Test GeminiRunner error handling for invalid/malformed output."""

    def test_parse_output_invalid_json(self):
        """Invalid JSON should raise ParseError with context."""
        runner = GeminiRunner()
        invalid_json = "not valid json {{"

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(invalid_json, stderr="")

        assert "Invalid JSON" in str(exc_info.value)
        assert exc_info.value.raw_output == invalid_json

    def test_parse_output_missing_response_field(self):
        """Missing 'response' field should raise ParseError."""
        runner = GeminiRunner()
        missing_field = '{"stats": {}}'  # No "response" key

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(missing_field, stderr="")

        assert "Missing 'response' field" in str(exc_info.value)
        assert exc_info.value.raw_output == missing_field

    def test_parse_output_non_string_response(self):
        """Non-string 'response' value should raise ParseError."""
        runner = GeminiRunner()
        non_string_response = '{"response": 123}'  # response should be string

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(non_string_response, stderr="")

        assert "'response' field must be a string" in str(exc_info.value)
        assert exc_info.value.raw_output == non_string_response


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
        runner = GeminiRunner()
        request = make_prompt_request(
            agent="gemini",
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
            stdout=-1,
            stderr=-1,
        )

        # Assert: Response parsed correctly
        assert response.agent == "gemini"
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
        runner = GeminiRunner()
        request = make_prompt_request()

        # Act & Assert
        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 1
        assert "API key not found" in exc_info.value.stderr


class TestGeminiRunnerCLIDetection:
    """Test GeminiRunner CLI detection at construction time."""

    def test_raises_if_cli_not_found(self, mock_cli_detection):
        """GeminiRunner should raise CLINotFoundError if gemini not in PATH."""
        mock_cli_detection.return_value = CLIInfo(found=False)
        with pytest.raises(CLINotFoundError):
            GeminiRunner()

    def test_uses_json_if_supported(self):
        """With version 0.12.0 (autouse fixture), JSON flag should be present."""
        runner = GeminiRunner()
        cmd = runner.build_command(make_prompt_request(prompt="Test"))
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_omits_json_if_not_supported(self):
        """With old version, JSON flag should be absent."""
        with (
            patch("nexus_mcp.runners.gemini.detect_cli") as mock_detect,
            patch("nexus_mcp.runners.gemini.get_cli_version", return_value="0.5.0"),
        ):
            mock_detect.return_value = CLIInfo(found=True, path="/usr/bin/gemini")
            runner = GeminiRunner()
        cmd = runner.build_command(make_prompt_request(prompt="Test"))
        assert "--output-format" not in cmd


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

        runner = GeminiRunner()
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

        runner = GeminiRunner()
        request = make_prompt_request(prompt="test")

        # Should raise - no recovery possible
        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 1
        assert "Error: 401 Unauthorized" in exc_info.value.stderr


class TestGeminiRunnerAPIErrorExtraction:
    """Test GeminiRunner extracts structured API error details from error JSON."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_error_json_surfaces_in_subprocess_error(self, mock_exec):
        """429 rate limit error JSON should produce a structured SubprocessError message."""
        error_stdout = (
            '{"error": {"code": 429, "message": "Quota exceeded for quota metric",'
            ' "status": "RESOURCE_EXHAUSTED"}}'
        )
        mock_exec.return_value = create_mock_process(
            stdout=error_stdout,
            stderr="",
            returncode=1,
        )
        runner = GeminiRunner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        # The primary message (args[0]) should contain structured error info,
        # not just raw JSON buried in stdout
        primary_message = exc_info.value.args[0]
        assert "429" in primary_message
        assert "RESOURCE_EXHAUSTED" in primary_message
        assert "Gemini API error" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_error_json_401_surfaces_auth_failure(self, mock_exec):
        """401 auth error JSON should produce a structured SubprocessError message."""
        error_stdout = (
            '{"error": {"code": 401, "message": "API key not valid", "status": "UNAUTHENTICATED"}}'
        )
        mock_exec.return_value = create_mock_process(
            stdout=error_stdout,
            stderr="",
            returncode=1,
        )
        runner = GeminiRunner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        primary_message = exc_info.value.args[0]
        assert "401" in primary_message
        assert "UNAUTHENTICATED" in primary_message
        assert "Gemini API error" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_error_json_preserves_returncode(self, mock_exec):
        """SubprocessError from error JSON should preserve returncode and stdout."""
        error_stdout = (
            '{"error": {"code": 429, "message": "Rate limited", "status": "RESOURCE_EXHAUSTED"}}'
        )
        mock_exec.return_value = create_mock_process(
            stdout=error_stdout,
            stderr="rate limit hit",
            returncode=1,
        )
        runner = GeminiRunner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 1
        assert exc_info.value.stdout == error_stdout

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_stderr_json_fallback_when_stdout_empty(self, mock_exec):
        """When stdout is empty but stderr has JSON error block, extracts structured error."""
        stderr_with_json = (
            "Gemini CLI error log\n"
            "Stack trace...\n"
            '{"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}}'
        )
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr=stderr_with_json,
            returncode=1,
        )
        runner = GeminiRunner()
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
        stdout_error = (
            '{"error": {"code": 429, "message": "Quota exceeded for stdout",'
            ' "status": "RESOURCE_EXHAUSTED"}}'
        )
        stderr_with_json = (
            "log line\n"
            '{"error": {"code": 1, "message": "Generic exit code error", "status": "UNKNOWN"}}'
        )
        mock_exec.return_value = create_mock_process(
            stdout=stdout_error,
            stderr=stderr_with_json,
            returncode=1,
        )
        runner = GeminiRunner()
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
        runner = GeminiRunner()
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
        runner = GeminiRunner()
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
        runner = GeminiRunner()
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
        error_stdout = (
            '{"error": {"code": 429, "message": "Rate limited", "status": "RESOURCE_EXHAUSTED"}}'
        )
        mock_exec.return_value = create_mock_process(
            stdout=error_stdout,
            stderr="",
            returncode=1,
        )
        runner = GeminiRunner()
        request = make_prompt_request()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        # The command field should be populated (not None), matching generic error behavior
        assert exc_info.value.command is not None
        assert "gemini" in exc_info.value.command[0]


class TestGeminiRunnerScalarJson:
    """Test parse_output handles scalar JSON values (null, number) without TypeError."""

    def test_parse_output_null_json_raises_parse_error(self):
        """json.loads('null') returns None; 'response' not in None raises TypeError without fix."""
        runner = GeminiRunner()

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output("null", stderr="")

        assert "Missing 'response' field" in str(exc_info.value)

    def test_parse_output_number_json_raises_parse_error(self):
        """json.loads('42') returns int; 'response' not in 42 raises TypeError without fix."""
        runner = GeminiRunner()

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output("42", stderr="")

        assert "Missing 'response' field" in str(exc_info.value)


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
        runner = GeminiRunner()
        request = make_prompt_request()

        response = await runner.run(request)

        assert response.output == "Partial answer"
        assert response.metadata.get("recovered_from_error") is True


class TestGeminiRunnerExtractLastJson:
    """Test GeminiRunner._extract_last_json_object() helper staticmethod."""

    def test_extracts_json_from_end_of_mixed_text(self):
        """Extracts the JSON object appended at the end of a mixed-content string."""
        text = 'log line 1\nlog line 2\n{"key": "value"}'
        result = GeminiRunner._extract_last_json_object(text)
        assert result == {"key": "value"}

    def test_returns_none_for_no_json(self):
        """Returns None when the text contains no JSON object."""
        result = GeminiRunner._extract_last_json_object("just plain text with no JSON")
        assert result is None

    def test_returns_none_for_empty_string(self):
        """Returns None for empty string input."""
        result = GeminiRunner._extract_last_json_object("")
        assert result is None

    def test_handles_nested_braces(self):
        """Correctly handles JSON with nested objects (brace-depth matching)."""
        text = 'preamble\n{"outer": {"inner": "value", "count": 42}}'
        result = GeminiRunner._extract_last_json_object(text)
        assert result == {"outer": {"inner": "value", "count": 42}}

    def test_picks_last_json_when_multiple_present(self):
        """Returns the last JSON object when multiple are present in the text."""
        text = '{"first": 1}\nsome text\n{"second": 2}'
        result = GeminiRunner._extract_last_json_object(text)
        assert result == {"second": 2}

    def test_ignores_non_dict_json(self):
        """Returns None when the extracted JSON is not a dict (e.g., an array)."""
        text = "text before\n[1, 2, 3]"
        result = GeminiRunner._extract_last_json_object(text)
        assert result is None

    def test_known_limitation_braces_in_string_values(self):
        """Documents known limitation: brace chars in string values confuse depth tracking.

        Balanced brace chars (e.g. '{foo}') cancel out and accidentally find the right
        boundary; unbalanced ones (e.g. lone '}') cause json.loads to fail and return None.
        In practice, Google API error messages rarely contain literal brace characters.
        """
        # Balanced braces in string values — algorithm accidentally gets the right answer
        text = 'log\n{"key": "{balanced}"}'
        result = GeminiRunner._extract_last_json_object(text)
        assert result == {"key": "{balanced}"}

        # Unbalanced closing brace in string value — algorithm mis-identifies boundary,
        # json.loads fails, returns None (falls through to generic SubprocessError)
        text_unbalanced = 'log\n{"key": "has a } brace"}'
        result_unbalanced = GeminiRunner._extract_last_json_object(text_unbalanced)
        # NOTE: Returns None due to brace-depth mis-parsing. Raw stderr is still
        # preserved in the generic SubprocessError at base.py:99-105.
        assert result_unbalanced is None


class TestGeminiRunnerFileReferences:
    """Test GeminiRunner file references handling."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_gemini_runner_appends_file_refs_to_prompt(self, mock_exec):
        """GeminiRunner appends file references to prompt text."""
        mock_exec.return_value = create_mock_process(stdout='{"response": "ok"}')

        runner = GeminiRunner()
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

        runner = GeminiRunner()
        request = make_prompt_request(prompt="Simple prompt")
        await runner.run(request)

        args = mock_exec.call_args[0]
        prompt_arg = args[2]
        assert prompt_arg == "Simple prompt"  # Unchanged


class TestGeminiRunnerNoisyStdout:
    """Test GeminiRunner.parse_output() handles Node.js warning prefix before JSON."""

    def test_parse_output_with_node_warnings(self):
        """Node.js deprecation warnings before valid JSON → should parse successfully."""
        runner = GeminiRunner()

        response = runner.parse_output(GEMINI_NOISY_STDOUT, stderr="")

        assert response.agent == "gemini"
        assert response.output == "test output"
        assert response.raw_output == GEMINI_NOISY_STDOUT

    def test_parse_output_with_multiple_prefix_lines(self):
        """Multiple non-JSON lines before JSON → should extract and parse correctly."""
        runner = GeminiRunner()
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
        runner = GeminiRunner()
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
        runner = GeminiRunner()
        noisy_no_json = (
            "(node:87799) [DEP0040] DeprecationWarning: punycode is deprecated\n"
            "Loaded cached credentials.\n"
            "Error: Something went wrong without JSON output"
        )

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(noisy_no_json, stderr="")

        assert exc_info.value.raw_output == noisy_no_json


class TestGeminiRunnerEnvConfiguration:
    """Test GeminiRunner environment variable configuration."""

    @patch.dict("os.environ", {"NEXUS_GEMINI_PATH": "/opt/custom/gemini"})
    def test_gemini_runner_uses_custom_path_from_env(self):
        """GeminiRunner uses NEXUS_GEMINI_PATH if set."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="test")

        # Build command and verify custom path used
        cmd = runner.build_command(request)
        assert cmd[0] == "/opt/custom/gemini"

    @patch.dict("os.environ", {"NEXUS_GEMINI_MODEL": "gemini-2.5-flash"})
    def test_gemini_runner_uses_default_model_from_env(self):
        """GeminiRunner uses NEXUS_GEMINI_MODEL as default if request.model is None."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="test", model=None)

        cmd = runner.build_command(request)
        assert "--model" in cmd
        assert "gemini-2.5-flash" in cmd

    @patch.dict("os.environ", {"NEXUS_GEMINI_MODEL": "gemini-2.5-flash"})
    def test_gemini_runner_request_model_overrides_env(self):
        """Request model overrides env var default."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="test", model="gemini-3-pro-preview")

        cmd = runner.build_command(request)
        assert "gemini-3-pro-preview" in cmd
        assert "gemini-2.5-flash" not in cmd
