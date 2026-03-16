# tests/unit/runners/test_opencode.py
"""Tests for OpenCodeRunner.

Tests verify:
- CLINotFoundError when opencode binary is absent
- build_command() constructs correct base argument list
- build_command() respects model override and env default
- build_command() appends file_refs to prompt
- execution mode flags: default/yolo (all produce same command structure)
- parse_output() NDJSON primary path and JSON fallback path
- error handling: _recover_from_error, _try_extract_error, retry integration
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.exceptions import CLINotFoundError, ParseError, RetryableError, SubprocessError
from nexus_mcp.runners.opencode import OpenCodeRunner
from tests.fixtures import (
    OPENCODE_JSON_RESPONSE,
    OPENCODE_NDJSON_RESPONSE,
    OPENCODE_NOISY_STDOUT,
    create_mock_process,
    make_prompt_request,
    opencode_error_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_opencode_runner() -> OpenCodeRunner:
    """Create an OpenCodeRunner using the autouse cli_detection_mocks fixture."""
    return OpenCodeRunner()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerInit:
    """Test OpenCodeRunner initialisation and CLI detection."""

    def test_cli_not_found_raises_error(self):
        """CLINotFoundError raised when opencode binary is absent."""
        not_found = CLIInfo(found=False, path=None)
        with (
            patch("nexus_mcp.runners.base.detect_cli", return_value=not_found),
            pytest.raises(CLINotFoundError) as exc_info,
        ):
            OpenCodeRunner()
        assert exc_info.value.cli_name == "opencode"

    def test_default_cli_path(self):
        """Default cli_path is 'opencode' when NEXUS_OPENCODE_PATH is not set."""
        runner = make_opencode_runner()
        assert runner.cli_path == "opencode"

    def test_custom_cli_path_from_env(self):
        """cli_path is taken from NEXUS_OPENCODE_PATH env var."""
        with patch(
            "nexus_mcp.runners.base.get_agent_env",
            side_effect=lambda _agent, key, **_kw: "/custom/opencode" if key == "PATH" else None,
        ):
            runner = make_opencode_runner()
        assert runner.cli_path == "/custom/opencode"

    def test_default_model_from_env(self):
        """default_model is taken from NEXUS_OPENCODE_MODEL env var."""
        with patch(
            "nexus_mcp.runners.base.get_agent_env",
            side_effect=lambda _agent, key, **_kw: "gpt-4o" if key == "MODEL" else None,
        ):
            runner = make_opencode_runner()
        assert runner.default_model == "gpt-4o"


# ---------------------------------------------------------------------------
# build_command — core shape
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerBuildCommand:
    """Test OpenCodeRunner.build_command() base argument construction."""

    def test_default_command_shape(self):
        """Default command: [cli_path, 'run', prompt, '--format', 'json', ...]."""
        runner = make_opencode_runner()
        cmd = runner.build_command(make_prompt_request(agent="opencode", prompt="hello"))
        assert cmd[0] == "opencode"
        assert cmd[1] == "run"
        assert cmd[2] == "hello"
        assert cmd[3] == "--format"
        assert cmd[4] == "json"

    def test_model_override(self):
        """request.model is forwarded as --model flag."""
        runner = make_opencode_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="opencode", prompt="hi", model="gpt-4o")
        )
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gpt-4o"

    def test_env_model_default(self):
        """default_model from env is used when request.model is None."""
        runner = make_opencode_runner()
        runner.default_model = "claude-3-5-sonnet"
        cmd = runner.build_command(make_prompt_request(agent="opencode", prompt="hi"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-3-5-sonnet"

    def test_request_model_overrides_env_default(self):
        """request.model takes precedence over default_model."""
        runner = make_opencode_runner()
        runner.default_model = "claude-3-5-sonnet"
        cmd = runner.build_command(
            make_prompt_request(agent="opencode", prompt="hi", model="gpt-4o")
        )
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gpt-4o"
        assert "claude-3-5-sonnet" not in cmd


# ---------------------------------------------------------------------------
# build_command — execution modes (inverted permission model)
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerBuildCommandModes:
    """Test OpenCodeRunner execution mode handling.

    OpenCode's `run` subcommand does not expose tool restriction flags.
    All execution modes produce the same base command structure.
    """

    def test_default_mode_produces_valid_command(self):
        """execution_mode='default' produces a valid command with --format json."""
        runner = make_opencode_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="opencode", prompt="x", execution_mode="default")
        )
        assert "run" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    def test_yolo_mode_produces_valid_command(self):
        """execution_mode='yolo' produces a valid command with --format json."""
        runner = make_opencode_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="opencode", prompt="x", execution_mode="yolo")
        )
        assert "run" in cmd
        assert "--format" in cmd
        assert "json" in cmd

    def test_no_tool_restriction_flags_in_any_mode(self):
        """No tool restriction flags are added (opencode run has no --allowedTools)."""
        runner = make_opencode_runner()
        for mode in ("default", "yolo"):
            cmd = runner.build_command(
                make_prompt_request(agent="opencode", prompt="x", execution_mode=mode)
            )
            assert "--allowedTools" not in cmd

    def test_model_with_default_mode(self):
        """model + default mode: --model present in command."""
        runner = make_opencode_runner()
        cmd = runner.build_command(
            make_prompt_request(
                agent="opencode", prompt="x", model="gpt-4o", execution_mode="default"
            )
        )
        assert "--model" in cmd

    def test_model_with_yolo_mode(self):
        """model + yolo mode: --model present in command."""
        runner = make_opencode_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="opencode", prompt="x", model="gpt-4o", execution_mode="yolo")
        )
        assert "--model" in cmd

    def test_model_after_format_flag(self):
        """--model appears after --format json in the command."""
        runner = make_opencode_runner()
        cmd = runner.build_command(
            make_prompt_request(
                agent="opencode", prompt="x", model="gpt-4o", execution_mode="default"
            )
        )
        assert cmd.index("--model") > cmd.index("--format")


# ---------------------------------------------------------------------------
# File references
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerFileReferences:
    """Test OpenCodeRunner file references handling in build_command()."""

    def test_file_refs_appended_to_prompt(self):
        """File references are appended to the prompt string."""
        runner = make_opencode_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="opencode", prompt="analyse", file_refs=["src/main.py"])
        )
        assert cmd[2].startswith("analyse")
        assert "src/main.py" in cmd[2]

    def test_prompt_unchanged_when_no_file_refs(self):
        """Prompt is unchanged when file_refs is empty."""
        runner = make_opencode_runner()
        cmd = runner.build_command(make_prompt_request(agent="opencode", prompt="hello"))
        assert cmd[2] == "hello"


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerParseOutput:
    """Test OpenCodeRunner.parse_output() — NDJSON primary and JSON fallback paths."""

    # NDJSON primary path

    def test_success_ndjson(self):
        """Valid NDJSON with agent_message → AgentResponse."""
        runner = make_opencode_runner()
        result = runner.parse_output(OPENCODE_NDJSON_RESPONSE, "")
        assert result.agent == "opencode"
        assert result.output == "test output"
        assert result.raw_output == OPENCODE_NDJSON_RESPONSE

    def test_ndjson_multiple_messages_joined(self):
        """Multiple text events are joined with \\n\\n."""

        def event(text: str) -> str:
            return json.dumps({"type": "text", "part": {"type": "text", "text": text}})

        ndjson = "\n".join([event("part1"), event("part2")])
        runner = make_opencode_runner()
        result = runner.parse_output(ndjson, "")
        assert result.output == "part1\n\npart2"

    def test_ndjson_strips_whitespace(self):
        """NDJSON output is stripped of leading/trailing whitespace."""
        ndjson = json.dumps({"type": "text", "part": {"type": "text", "text": "  pong  "}})
        runner = make_opencode_runner()
        result = runner.parse_output(ndjson, "")
        assert result.output == "pong"

    def test_ndjson_no_text_events_returns_empty_output(self):
        """Valid NDJSON with only non-text events → AgentResponse with empty output."""
        ndjson = "\n".join(
            [
                '{"type": "step_start", "sessionID": "ses_test"}',
                '{"type": "step_finish", "sessionID": "ses_test"}',
            ]
        )
        runner = make_opencode_runner()
        result = runner.parse_output(ndjson, "")
        assert result.agent == "opencode"
        assert result.output == ""
        assert result.raw_output == ndjson

    def test_ndjson_typeless_dicts_fall_through_to_json_fallback(self):
        """JSON dicts without 'type' key skip the NDJSON sentinel; fallback handles them."""
        # Typeless dict WITH a matching key → JSON fallback succeeds
        stdout = json.dumps({"message": "from typeless dict"})
        runner = make_opencode_runner()
        result = runner.parse_output(stdout, "")
        assert result.output == "from typeless dict"

    def test_ndjson_error_only_raises_parse_error(self):
        """NDJSON with only error events → ParseError (error skipped, no sentinel set)."""
        stdout = opencode_error_json("ServiceError", "overloaded", status_code=503)
        runner = make_opencode_runner()
        with pytest.raises(ParseError):
            runner.parse_output(stdout, "")

    # JSON object fallback path

    def test_success_json_message_key(self):
        """JSON object with 'message' key → AgentResponse via fallback."""
        runner = make_opencode_runner()
        result = runner.parse_output(OPENCODE_JSON_RESPONSE, "")
        assert result.agent == "opencode"
        assert result.output == "test output"

    def test_success_json_content_key(self):
        """JSON object with 'content' key → extracted."""
        runner = make_opencode_runner()
        stdout = json.dumps({"content": "content output"})
        result = runner.parse_output(stdout, "")
        assert result.output == "content output"

    def test_success_json_text_key(self):
        """JSON object with 'text' key → extracted."""
        runner = make_opencode_runner()
        stdout = json.dumps({"text": "text output"})
        result = runner.parse_output(stdout, "")
        assert result.output == "text output"

    def test_success_json_response_key(self):
        """JSON object with 'response' key → extracted."""
        runner = make_opencode_runner()
        stdout = json.dumps({"response": "response output"})
        result = runner.parse_output(stdout, "")
        assert result.output == "response output"

    def test_json_key_priority_message_over_text(self):
        """'message' key takes priority over 'text' when both present."""
        runner = make_opencode_runner()
        stdout = json.dumps({"message": "from message", "text": "from text"})
        result = runner.parse_output(stdout, "")
        assert result.output == "from message"

    def test_no_matching_key_raises_parse_error(self):
        """JSON with no matching key (not message/content/text/response) → ParseError."""
        runner = make_opencode_runner()
        stdout = json.dumps({"unknown_key": "value"})
        with pytest.raises(ParseError):
            runner.parse_output(stdout, "")

    def test_both_paths_fail_raises_parse_error(self):
        """Empty stdout → ParseError (neither path yields output)."""
        runner = make_opencode_runner()
        with pytest.raises(ParseError):
            runner.parse_output("", "")

    def test_plain_text_raises_parse_error(self):
        """Plain text (not JSON, not NDJSON) → ParseError."""
        runner = make_opencode_runner()
        with pytest.raises(ParseError):
            runner.parse_output("just some text", "")

    def test_agent_name_is_opencode(self):
        """result.agent is always 'opencode'."""
        runner = make_opencode_runner()
        result = runner.parse_output(OPENCODE_NDJSON_RESPONSE, "")
        assert result.agent == "opencode"

    def test_raw_output_is_original_stdout(self):
        """result.raw_output matches the original stdout passed in."""
        runner = make_opencode_runner()
        result = runner.parse_output(OPENCODE_NDJSON_RESPONSE, "some stderr")
        assert result.raw_output == OPENCODE_NDJSON_RESPONSE


# ---------------------------------------------------------------------------
# parse_output — edge cases (scalar JSON)
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerParseOutputEdgeCases:
    """Test parse_output handles scalar JSON values and non-dict JSON without crashing."""

    def test_json_array_falls_through_to_extract_fallback(self):
        """JSON array containing object with 'message' key → extracted via fallback."""
        stdout = json.dumps([{"message": "from array"}])
        runner = make_opencode_runner()
        result = runner.parse_output(stdout, "")
        assert result.output == "from array"

    def test_parse_output_null_raises_parse_error(self):
        """'null' → neither path yields output → ParseError."""
        runner = make_opencode_runner()
        with pytest.raises(ParseError):
            runner.parse_output("null", "")

    def test_parse_output_number_raises_parse_error(self):
        """'42' → neither path yields output → ParseError."""
        runner = make_opencode_runner()
        with pytest.raises(ParseError):
            runner.parse_output("42", "")

    def test_parse_output_bool_raises_parse_error(self):
        """'true' → neither path yields output → ParseError."""
        runner = make_opencode_runner()
        with pytest.raises(ParseError):
            runner.parse_output("true", "")


# ---------------------------------------------------------------------------
# Noisy stdout
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerNoisyStdout:
    """Test OpenCodeRunner.parse_output() handles prefix lines before valid output."""

    def test_noisy_stdout_json_fallback(self):
        """Log lines before JSON object are tolerated via extract_last_json_object."""
        runner = make_opencode_runner()
        stdout = "Loading...\nConnecting...\n" + json.dumps({"message": "actual output"})
        result = runner.parse_output(stdout, "")
        assert result.output == "actual output"

    def test_noisy_ndjson_with_prefix_lines(self):
        """Log lines before valid NDJSON → parsed via primary NDJSON path."""
        runner = make_opencode_runner()
        result = runner.parse_output(OPENCODE_NOISY_STDOUT, "")
        assert result.output == "test output"

    def test_noisy_stdout_no_json_raises_parse_error(self):
        """Log lines with no valid JSON → ParseError with raw_output preserved."""
        noisy_no_json = (
            "Loading configuration...\n"
            "Connecting to API...\n"
            "Error: Something went wrong without JSON output"
        )
        runner = make_opencode_runner()
        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(noisy_no_json, "")
        assert exc_info.value.raw_output == noisy_no_json


# ---------------------------------------------------------------------------
# Error handling (_recover_from_error and _try_extract_error unit tests)
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerErrorHandling:
    """Test OpenCodeRunner._recover_from_error and _try_extract_error."""

    def test_no_json_in_stderr_or_stdout_returns_none(self):
        """No structured error JSON → _recover_from_error returns None."""
        runner = make_opencode_runner()
        result = runner._recover_from_error("not json", "also not json", 1)
        assert result is None

    def test_valid_ndjson_on_error_returns_recovered_response(self):
        """Valid NDJSON in stdout on non-zero exit → recovered AgentResponse."""
        runner = make_opencode_runner()
        result = runner._recover_from_error(OPENCODE_NDJSON_RESPONSE, "", 1)
        assert result is not None
        assert result.output == "test output"
        assert result.metadata.get("recovered_from_error") is True

    def test_valid_json_on_error_returns_recovered_response(self):
        """Valid JSON object in stdout on non-zero exit → recovered AgentResponse."""
        runner = make_opencode_runner()
        result = runner._recover_from_error(OPENCODE_JSON_RESPONSE, "", 1)
        assert result is not None
        assert result.output == "test output"
        assert result.metadata.get("recovered_from_error") is True

    def test_malformed_json_in_stderr_returns_none(self):
        """Truncated/malformed JSON in stderr → _try_extract_error returns without raising."""
        stderr = '{"error": {"code": 429, "message": "truncated...'
        runner = make_opencode_runner()
        result = runner._recover_from_error("", stderr, 1)
        assert result is None

    def test_try_extract_error_stdout_fallback(self):
        """stderr="" + stdout has NDJSON error event with code 503 → RetryableError raised."""
        stdout = opencode_error_json("ServiceError", "unavailable", status_code=503)
        runner = make_opencode_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error(stdout, "", 1)

    def test_stderr_non_error_json_falls_through_to_stdout_ndjson(self):
        """stderr has unrelated JSON; stdout has NDJSON error event → error found in stdout."""
        stderr = json.dumps({"debug": "diagnostic info"})
        stdout = opencode_error_json("ServiceError", "overloaded", status_code=503)
        runner = make_opencode_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error(stdout, stderr, 1)

    def test_legacy_error_format_in_stderr_still_works(self):
        """Legacy {"error":{"code":503,...}} in stderr → RetryableError via fallback path."""
        stderr = json.dumps({"error": {"code": 503, "message": "Service unavailable"}})
        runner = make_opencode_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error("", stderr, 1)


# ---------------------------------------------------------------------------
# Retryable errors
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerRetryableErrors:
    """Test OpenCodeRunner retryable error classification and retry integration."""

    @pytest.mark.parametrize("code", [429, 503])
    def test_retryable_error_codes_raise_retryable_error(self, code: int):
        """Error codes 429 and 503 → RetryableError via NDJSON error event in stdout."""
        stdout = opencode_error_json("TransientError", "transient error", status_code=code)
        runner = make_opencode_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error(stdout, "", 1)

    def test_non_retryable_error_code_raises_subprocess_error(self):
        """Non-retryable error code → SubprocessError (not RetryableError)."""
        stdout = opencode_error_json("AuthError", "unauthorized", status_code=401)
        runner = make_opencode_runner()
        with pytest.raises(SubprocessError) as exc_info:
            runner._try_extract_error(stdout, "", 1)
        assert not isinstance(exc_info.value, RetryableError)

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_on_503_full_loop(self, mock_exec):
        """503 on first attempt, success on second → called twice."""
        error_stdout = opencode_error_json("ServiceError", "unavailable", status_code=503)
        mock_exec.side_effect = [
            create_mock_process(stdout=error_stdout, stderr="", returncode=1),
            create_mock_process(stdout=OPENCODE_NDJSON_RESPONSE, returncode=0),
        ]
        runner = make_opencode_runner()
        result = await runner.run(
            make_prompt_request(agent="opencode", prompt="test", max_retries=2)
        )
        assert result.output == "test output"
        assert mock_exec.await_count == 2

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_exhausted_all_503(self, mock_exec):
        """3x 503 → RetryableError raised, called 3 times."""
        error_stdout = opencode_error_json("ServiceError", "unavailable", status_code=503)
        mock_exec.return_value = create_mock_process(stdout=error_stdout, stderr="", returncode=1)
        runner = make_opencode_runner()
        request = make_prompt_request(agent="opencode", prompt="test", max_retries=3)
        with pytest.raises(RetryableError):
            await runner.run(request)
        assert mock_exec.await_count == 3


# ---------------------------------------------------------------------------
# Error recovery (full run() with mocked subprocess)
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerErrorRecovery:
    """Test OpenCodeRunner error recovery from non-zero exit codes."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_recovers_from_valid_ndjson_on_error_exit(self, mock_exec):
        """Non-zero exit + valid NDJSON stdout → recovered response with metadata stamped."""
        mock_exec.return_value = create_mock_process(
            stdout=OPENCODE_NDJSON_RESPONSE,
            stderr="Warning: something minor",
            returncode=1,
        )
        runner = make_opencode_runner()
        response = await runner.run(make_prompt_request(agent="opencode", prompt="test"))
        assert response.output == "test output"
        assert response.metadata.get("recovered_from_error") is True
        assert response.metadata.get("original_exit_code") == 1
        assert response.metadata.get("stderr") == "Warning: something minor"

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_raises_when_stdout_not_parseable(self, mock_exec):
        """Non-zero exit with no recoverable content → SubprocessError raised by run()."""
        mock_exec.return_value = create_mock_process(
            stdout="", stderr="something went wrong", returncode=1
        )
        runner = make_opencode_runner()
        with pytest.raises(SubprocessError):
            await runner.run(make_prompt_request(agent="opencode", prompt="x"))


# ---------------------------------------------------------------------------
# API error extraction (structured JSON in stderr/stdout)
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerAPIErrorExtraction:
    """Test OpenCodeRunner extracts structured API error details from error JSON."""

    @pytest.mark.parametrize(
        ("code", "message"),
        [
            (429, "rate limited"),
            (401, "unauthorized"),
        ],
        ids=["429-retryable", "401-auth"],
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_opencode_error_json_surfaces_in_subprocess_error(self, mock_exec, code, message):
        """stdout error NDJSON → SubprocessError with 'OpenCode API error' and error name."""
        mock_exec.return_value = create_mock_process(
            stdout=opencode_error_json("ApiError", message, status_code=code),
            stderr="",
            returncode=1,
        )
        runner = make_opencode_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="opencode", prompt="x", max_retries=1))

        primary_message = exc_info.value.args[0]
        assert "OpenCode API error" in primary_message
        assert "ApiError" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_structured_error_includes_command(self, mock_exec):
        """SubprocessError from structured error NDJSON includes the CLI command."""
        mock_exec.return_value = create_mock_process(
            stdout=opencode_error_json("AuthError", "unauthorized", status_code=401),
            stderr="",
            returncode=1,
        )
        runner = make_opencode_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="opencode", prompt="x", max_retries=1))

        assert exc_info.value.command is not None
        assert "opencode" in exc_info.value.command[0]

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_no_json_anywhere_falls_through_to_generic(self, mock_exec):
        """No JSON in stdout or stderr → generic SubprocessError without 'OpenCode API error'."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr="Connection refused: unable to reach API",
            returncode=1,
        )
        runner = make_opencode_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="opencode", prompt="x"))

        primary_message = exc_info.value.args[0]
        assert "OpenCode API error" not in primary_message
        assert "CLI command failed" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_non_dict_json_stdout_does_not_crash(self, mock_exec):
        """Array of scalars in stdout → generic SubprocessError (no AttributeError)."""
        mock_exec.return_value = create_mock_process(
            stdout="[1, 2, 3]",
            stderr="something went wrong",
            returncode=1,
        )
        runner = make_opencode_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="opencode", prompt="x"))

        primary_message = exc_info.value.args[0]
        assert "OpenCode API error" not in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_malformed_stderr_json_falls_through(self, mock_exec):
        """Truncated/malformed JSON in stderr → falls through to generic SubprocessError."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr='{"error": {"code": 429, "message": "truncated...',
            returncode=1,
        )
        runner = make_opencode_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="opencode", prompt="x"))

        primary_message = exc_info.value.args[0]
        assert "OpenCode API error" not in primary_message


# ---------------------------------------------------------------------------
# Dual-field recovery (valid NDJSON stdout wins over error stderr)
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerDualFieldRecovery:
    """Test _recover_from_error with NDJSON error events in stdout."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_error_only_stdout_raises_retryable_error(self, mock_exec):
        """stdout=error NDJSON event only, returncode=1 → RetryableError raised.

        Error events are skipped by the NDJSON parser (no sentinel set) so both
        parse paths fail. _try_extract_error then scans stdout and extracts the
        retryable error via statusCode=429.
        """
        mock_exec.return_value = create_mock_process(
            stdout=opencode_error_json("RateLimitError", "rate limited", status_code=429),
            stderr="",
            returncode=1,
        )
        runner = make_opencode_runner()
        request = make_prompt_request(agent="opencode", prompt="x")

        with pytest.raises(RetryableError):
            await runner.run(request)


# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerEnvConfiguration:
    """Test OpenCodeRunner environment variable configuration."""

    @patch.dict("os.environ", {"NEXUS_OPENCODE_PATH": "/opt/custom/opencode"})
    def test_opencode_runner_uses_custom_path_from_env(self):
        """OpenCodeRunner uses NEXUS_OPENCODE_PATH if set."""
        runner = make_opencode_runner()
        cmd = runner.build_command(make_prompt_request(agent="opencode", prompt="test"))
        assert cmd[0] == "/opt/custom/opencode"

    @patch.dict("os.environ", {"NEXUS_OPENCODE_MODEL": "gpt-4o"})
    def test_opencode_runner_uses_default_model_from_env(self):
        """OpenCodeRunner uses NEXUS_OPENCODE_MODEL as default if request.model is None."""
        runner = make_opencode_runner()
        cmd = runner.build_command(make_prompt_request(agent="opencode", prompt="test", model=None))
        assert "--model" in cmd
        assert "gpt-4o" in cmd

    @patch.dict("os.environ", {"NEXUS_OPENCODE_MODEL": "gpt-4o"})
    def test_opencode_runner_request_model_overrides_env(self):
        """Request model overrides NEXUS_OPENCODE_MODEL env default."""
        runner = make_opencode_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="opencode", prompt="test", model="claude-sonnet-4-6")
        )
        assert "claude-sonnet-4-6" in cmd
        assert "gpt-4o" not in cmd


# ---------------------------------------------------------------------------
# Integration (full run() with mocked subprocess)
# ---------------------------------------------------------------------------


class TestOpenCodeRunnerIntegration:
    """Test OpenCodeRunner.run() end-to-end with mocked subprocess."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_success_returns_parsed_response(self, mock_exec):
        """Successful run executes CLI with correct args and returns parsed response."""
        mock_exec.return_value = create_mock_process(
            stdout=OPENCODE_NDJSON_RESPONSE,
            returncode=0,
        )
        runner = make_opencode_runner()
        request = make_prompt_request(agent="opencode", prompt="test prompt")

        response = await runner.run(request)

        mock_exec.assert_awaited_once_with(
            "opencode",
            "run",
            "test prompt",
            "--format",
            "json",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert response.agent == "opencode"
        assert response.output == "test output"

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_subprocess_error_propagates(self, mock_exec):
        """Non-zero exit with no recoverable JSON → SubprocessError with returncode and stderr."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr="API key not found",
            returncode=1,
        )
        runner = make_opencode_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="opencode", prompt="x"))

        assert exc_info.value.returncode == 1
        assert "API key" in exc_info.value.stderr
