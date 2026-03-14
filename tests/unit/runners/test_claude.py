# tests/unit/runners/test_claude.py
"""Tests for ClaudeRunner.

Tests verify:
- CLINotFoundError when claude binary is absent
- build_command() constructs correct argument list
- build_command() respects model override and env default
- build_command() uses custom CLI path
- build_command() appends file_refs to prompt
- execution mode flags: yolo, sandbox (maps to default), default
- parse_output() success and failure paths, including is_error handling
- error handling: _recover_from_error, _try_extract_error
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.exceptions import CLINotFoundError, ParseError, RetryableError, SubprocessError
from nexus_mcp.runners.claude import ClaudeRunner
from tests.fixtures import (
    CLAUDE_JSON_RESPONSE,
    CLAUDE_NOISY_STDOUT,
    claude_error_json,
    claude_json,
    create_mock_process,
    make_prompt_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_claude_runner() -> ClaudeRunner:
    """Create a ClaudeRunner using the autouse cli_detection_mocks fixture."""
    return ClaudeRunner()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestClaudeRunnerInit:
    """Test ClaudeRunner initialisation and CLI detection."""

    def test_cli_not_found_raises_error(self):
        """CLINotFoundError raised when claude binary is absent."""
        not_found = CLIInfo(found=False, path=None)
        with (
            patch("nexus_mcp.runners.base.detect_cli", return_value=not_found),
            pytest.raises(CLINotFoundError) as exc_info,
        ):
            ClaudeRunner()
        assert exc_info.value.cli_name == "claude"

    def test_default_cli_path(self):
        """Default cli_path is 'claude' when NEXUS_CLAUDE_PATH is not set."""
        runner = make_claude_runner()
        assert runner.cli_path == "claude"

    def test_custom_cli_path_from_env(self):
        """cli_path is taken from NEXUS_CLAUDE_PATH env var."""
        with patch(
            "nexus_mcp.runners.base.get_agent_env",
            side_effect=lambda _agent, key, **_kw: "/custom/claude" if key == "PATH" else None,
        ):
            runner = make_claude_runner()
        assert runner.cli_path == "/custom/claude"


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------


class TestClaudeRunnerBuildCommand:
    """Test ClaudeRunner.build_command() argument construction."""

    def test_default_command_shape(self):
        """Default command: [cli_path, '-p', prompt, '--output-format', 'json']."""
        runner = make_claude_runner()
        cmd = runner.build_command(make_prompt_request(agent="claude", prompt="hello"))
        assert cmd == ["claude", "-p", "hello", "--output-format", "json"]

    def test_model_override(self):
        """request.model is forwarded as --model flag."""
        runner = make_claude_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="claude", prompt="hi", model="claude-sonnet-4-6")
        )
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4-6"

    def test_env_model_default(self):
        """default_model from env is used when request.model is None."""
        runner = make_claude_runner()
        runner.default_model = "claude-haiku-4-5-20251001"
        cmd = runner.build_command(make_prompt_request(agent="claude", prompt="hi"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-haiku-4-5-20251001"

    def test_request_model_overrides_env_default(self):
        """request.model takes precedence over default_model."""
        runner = make_claude_runner()
        runner.default_model = "claude-haiku-4-5-20251001"
        cmd = runner.build_command(
            make_prompt_request(agent="claude", prompt="hi", model="claude-sonnet-4-6")
        )
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4-6"
        assert "claude-haiku-4-5-20251001" not in cmd

    def test_all_options_combined(self):
        """model + yolo + file_refs: all options present in correct order."""
        runner = make_claude_runner()
        cmd = runner.build_command(
            make_prompt_request(
                agent="claude",
                prompt="analyse",
                model="claude-sonnet-4-6",
                execution_mode="yolo",
                file_refs=["src/main.py"],
            )
        )
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert cmd.index("--model") < cmd.index("--dangerously-skip-permissions")


# ---------------------------------------------------------------------------
# build_command — execution modes
# ---------------------------------------------------------------------------


class TestClaudeRunnerBuildCommandModes:
    """Test ClaudeRunner execution mode flags in build_command()."""

    def test_yolo_mode_adds_flag(self):
        """execution_mode='yolo' adds --dangerously-skip-permissions flag."""
        runner = make_claude_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="claude", prompt="x", execution_mode="yolo")
        )
        assert "--dangerously-skip-permissions" in cmd

    def test_sandbox_mode_maps_to_default(self):
        """execution_mode='sandbox' adds no extra flags (Claude has no sandbox)."""
        runner = make_claude_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="claude", prompt="x", execution_mode="sandbox")
        )
        assert "--dangerously-skip-permissions" not in cmd

    def test_default_mode_no_extra_flags(self):
        """execution_mode='default' adds no extra approve flags."""
        runner = make_claude_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="claude", prompt="x", execution_mode="default")
        )
        assert "--dangerously-skip-permissions" not in cmd


# ---------------------------------------------------------------------------
# File references
# ---------------------------------------------------------------------------


class TestClaudeRunnerFileReferences:
    """Test ClaudeRunner file references handling in build_command()."""

    def test_file_refs_appended_to_prompt(self):
        """File references are appended to the prompt string."""
        runner = make_claude_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="claude", prompt="analyse", file_refs=["src/main.py"])
        )
        assert cmd[2].startswith("analyse")
        assert "src/main.py" in cmd[2]

    def test_prompt_unchanged_when_no_file_refs(self):
        """Prompt is unchanged when file_refs is empty."""
        runner = make_claude_runner()
        cmd = runner.build_command(make_prompt_request(agent="claude", prompt="hello"))
        assert cmd[2] == "hello"


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------


class TestClaudeRunnerParseOutput:
    """Test ClaudeRunner.parse_output()."""

    def test_success_extracts_result_text(self):
        """Valid Claude JSON array → AgentResponse with correct output."""
        runner = make_claude_runner()
        result = runner.parse_output(CLAUDE_JSON_RESPONSE, "")
        assert result.agent == "claude"
        assert result.output == "test output"
        assert result.raw_output == CLAUDE_JSON_RESPONSE

    def test_metadata_extracted(self):
        """Metadata contains cost_usd, duration_ms, num_turns, session_id."""
        runner = make_claude_runner()
        result = runner.parse_output(CLAUDE_JSON_RESPONSE, "")
        assert "cost_usd" in result.metadata
        assert "duration_ms" in result.metadata
        assert "num_turns" in result.metadata
        assert "session_id" in result.metadata

    def test_fallback_to_assistant_content(self):
        """JSON array with no 'result' type but has 'assistant' → extracts from content."""
        data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "assistant response"}],
                },
                "cost_usd": 0.001,
                "duration_ms": 1000,
            }
        ]
        runner = make_claude_runner()
        result = runner.parse_output(json.dumps(data), "")
        assert result.output == "assistant response"

    def test_fallback_metadata_from_assistant(self):
        """Assistant-fallback path includes cost_usd and duration_ms in metadata."""
        data = [
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                "cost_usd": 0.002,
                "duration_ms": 2000,
            }
        ]
        runner = make_claude_runner()
        result = runner.parse_output(json.dumps(data), "")
        assert result.metadata.get("cost_usd") == 0.002
        assert result.metadata.get("duration_ms") == 2000

    def test_whitespace_stripping(self):
        """Output text is stripped of leading/trailing whitespace."""
        data = [{"type": "result", "result": "  hello world  "}]
        runner = make_claude_runner()
        result = runner.parse_output(json.dumps(data), "")
        assert result.output == "hello world"

    def test_empty_stdout_raises_parse_error(self):
        """Empty string → ParseError."""
        runner = make_claude_runner()
        with pytest.raises(ParseError):
            runner.parse_output("", "")

    def test_invalid_json_raises_parse_error(self):
        """Non-JSON string → ParseError."""
        runner = make_claude_runner()
        with pytest.raises(ParseError):
            runner.parse_output("this is not json", "")

    def test_no_result_or_assistant_raises_parse_error(self):
        """Array with only irrelevant types → ParseError."""
        data = [{"type": "system"}, {"type": "tool_result"}]
        runner = make_claude_runner()
        with pytest.raises(ParseError):
            runner.parse_output(json.dumps(data), "")

    def test_non_string_result_raises_parse_error(self):
        """Non-string 'result' field → ParseError (explicit type check)."""
        data = [{"type": "result", "result": 42}]
        runner = make_claude_runner()
        with pytest.raises(ParseError):
            runner.parse_output(json.dumps(data), "")

    def test_noisy_stdout_with_log_prefix(self):
        """Log lines before JSON array still parses via extract_last_json_list fallback."""
        noisy = "(node:1234) DeprecationWarning: some warning\nLoaded credentials.\n" + claude_json(
            "pong from noisy output"
        )
        runner = make_claude_runner()
        result = runner.parse_output(noisy, "")
        assert result.output == "pong from noisy output"

    def test_non_list_json_raises_parse_error(self):
        """Non-object, non-array JSON (e.g. bare integer) → ParseError."""
        runner = make_claude_runner()
        with pytest.raises(ParseError):
            runner.parse_output("42", "")

    def test_single_object_result_parsed(self):
        """Single JSON object with type=result → parsed correctly (new CLI format)."""
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "pong",
                "duration_ms": 2492,
                "session_id": "sess-new",
                "cost_usd": 0.003,
                "num_turns": 1,
            }
        )
        runner = make_claude_runner()
        result = runner.parse_output(stdout, "")
        assert result.output == "pong"
        assert result.agent == "claude"

    def test_single_object_with_metadata(self):
        """Single JSON object → metadata fields extracted correctly."""
        stdout = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "hello",
                "duration_ms": 1234,
                "session_id": "sess-abc",
                "cost_usd": 0.001,
                "num_turns": 2,
            }
        )
        runner = make_claude_runner()
        result = runner.parse_output(stdout, "")
        assert result.metadata.get("duration_ms") == 1234
        assert result.metadata.get("session_id") == "sess-abc"
        assert result.metadata.get("cost_usd") == 0.001
        assert result.metadata.get("num_turns") == 2

    def test_is_error_result_raises_parse_error(self):
        """Result element with is_error=true → ParseError (not successful output)."""
        data = [{"type": "result", "is_error": True, "result": "Error: something failed"}]
        runner = make_claude_runner()
        with pytest.raises(ParseError, match="error result"):
            runner.parse_output(json.dumps(data), "")

    def test_preserves_internal_whitespace(self):
        """Result text: internal newlines preserved, only leading/trailing whitespace stripped."""
        data = [{"type": "result", "result": "Line 1\n\nLine 2"}]
        runner = make_claude_runner()
        result = runner.parse_output(json.dumps(data), "")
        assert result.output == "Line 1\n\nLine 2"

    def test_empty_result_string_is_valid(self):
        """Empty string result is valid; parse_output does not guard against empty strings."""
        data = [{"type": "result", "result": ""}]
        runner = make_claude_runner()
        result = runner.parse_output(json.dumps(data), "")
        assert result.output == ""


# ---------------------------------------------------------------------------
# parse_output — advanced
# ---------------------------------------------------------------------------


class TestClaudeRunnerParseOutputAdvanced:
    """Test advanced parse_output() behaviors for ClaudeRunner."""

    @pytest.fixture
    def runner(self) -> ClaudeRunner:
        return make_claude_runner()

    def test_multiple_result_elements_last_wins(self, runner):
        """Two type='result' elements → last one used (reversed() iteration)."""
        data = [
            {"type": "result", "result": "first result"},
            {"type": "result", "result": "last result"},
        ]
        result = runner.parse_output(json.dumps(data), "")
        assert result.output == "last result"

    def test_multi_block_assistant_text_joined(self, runner):
        """Two {'type':'text'} blocks in assistant content → joined with '\\n\\n'."""
        data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "block one"},
                        {"type": "text", "text": "block two"},
                    ],
                },
                "cost_usd": 0.001,
                "duration_ms": 1000,
            }
        ]
        result = runner.parse_output(json.dumps(data), "")
        assert result.output == "block one\n\nblock two"

    def test_non_text_blocks_skipped(self, runner):
        """tool_use block + text block → only text block included in output."""
        data = [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}},
                        {"type": "text", "text": "only this text"},
                    ],
                },
                "cost_usd": 0.001,
                "duration_ms": 1000,
            }
        ]
        result = runner.parse_output(json.dumps(data), "")
        assert result.output == "only this text"

    def test_noisy_stdout_object_fallback(self, runner):
        """Log lines + single JSON object → extract_last_json_object path used."""
        single_obj = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "object fallback result",
                "duration_ms": 1000,
                "session_id": "sess-fallback",
                "cost_usd": 0.001,
                "num_turns": 1,
            }
        )
        noisy = "Loading configuration...\nConnecting to API...\n" + single_obj
        result = runner.parse_output(noisy, "")
        assert result.output == "object fallback result"

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_recovered_response_metadata_fields(self, mock_exec):
        """On nonzero exit with parseable stdout, original_exit_code and stderr present."""
        mock_exec.return_value = create_mock_process(
            stdout=CLAUDE_JSON_RESPONSE,
            stderr="some stderr text",
            returncode=1,
        )
        runner = make_claude_runner()
        result = await runner.run(make_prompt_request(agent="claude", prompt="test"))
        assert result.metadata.get("original_exit_code") == 1
        assert result.metadata.get("stderr") == "some stderr text"
        assert result.metadata.get("recovered_from_error") is True

    def test_is_error_without_result_key(self, runner):
        """is_error=True, no 'result' key → ParseError with 'unknown error'."""
        data = [{"type": "result", "is_error": True}]
        with pytest.raises(ParseError, match="unknown error"):
            runner.parse_output(json.dumps(data), "")

    def test_is_error_with_none_result_uses_fallback_message(self, runner):
        """is_error=True, result=None → ParseError with 'unknown error' (None coerced)."""
        data = [{"type": "result", "is_error": True, "result": None}]
        with pytest.raises(ParseError, match="unknown error"):
            runner.parse_output(json.dumps(data), "")

    def test_assistant_with_non_dict_message(self, runner):
        """Non-dict 'message' value → content defaults to [] → no parts → ParseError."""
        data = [
            {"type": "assistant", "message": "not a dict", "cost_usd": 0.001, "duration_ms": 500}
        ]
        with pytest.raises(ParseError):
            runner.parse_output(json.dumps(data), "")

    def test_assistant_with_non_list_content(self, runner):
        """Non-list 'content' in assistant message → skips text loop → ParseError."""
        data = [
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": "not a list"},
                "cost_usd": 0.001,
                "duration_ms": 500,
            }
        ]
        with pytest.raises(ParseError):
            runner.parse_output(json.dumps(data), "")


# ---------------------------------------------------------------------------
# Scalar JSON (null, bool) edge cases
# ---------------------------------------------------------------------------


class TestClaudeRunnerScalarJson:
    """Test parse_output handles scalar JSON values (null, bool) without TypeError."""

    def test_parse_output_null_json_raises_parse_error(self):
        """json.loads('null') → None; not a dict/list → ParseError (not TypeError)."""
        runner = make_claude_runner()
        with pytest.raises(ParseError):
            runner.parse_output("null", "")

    def test_parse_output_boolean_json_raises_parse_error(self):
        """json.loads('true') → True; not a dict/list → ParseError (not TypeError)."""
        runner = make_claude_runner()
        with pytest.raises(ParseError):
            runner.parse_output("true", "")


# ---------------------------------------------------------------------------
# Noisy stdout (log prefix before JSON array)
# ---------------------------------------------------------------------------


class TestClaudeRunnerNoisyStdout:
    """Test ClaudeRunner.parse_output() handles log prefix before JSON array."""

    def test_parse_output_with_claude_noisy_stdout_fixture(self):
        """CLAUDE_NOISY_STDOUT constant with log prefix → parses via json_list fallback."""
        runner = make_claude_runner()
        result = runner.parse_output(CLAUDE_NOISY_STDOUT, "")
        assert result.agent == "claude"
        assert result.output == "test output"
        assert result.raw_output == CLAUDE_NOISY_STDOUT

    def test_noisy_stdout_preserves_metadata(self):
        """Noisy prefix + claude_json with custom metadata → metadata fields extracted correctly."""
        noisy = "Loading configuration...\nConnecting to API...\n" + claude_json(
            "pong", cost_usd=0.01, duration_ms=3000, num_turns=2
        )
        runner = make_claude_runner()
        result = runner.parse_output(noisy, "")
        assert result.output == "pong"
        assert result.metadata.get("cost_usd") == 0.01
        assert result.metadata.get("duration_ms") == 3000
        assert result.metadata.get("num_turns") == 2

    def test_noisy_stdout_no_json_raises_parse_error(self):
        """Log lines with no valid JSON → ParseError with raw_output preserved."""
        noisy_no_json = (
            "Loading configuration...\n"
            "Connecting to API...\n"
            "Error: Something went wrong without JSON output"
        )
        runner = make_claude_runner()
        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(noisy_no_json, "")
        assert exc_info.value.raw_output == noisy_no_json


# ---------------------------------------------------------------------------
# Error handling (_recover_from_error and _try_extract_error unit tests)
# ---------------------------------------------------------------------------


class TestClaudeRunnerErrorHandling:
    """Test ClaudeRunner._recover_from_error and _try_extract_error."""

    def test_no_json_in_stderr_or_stdout_returns_none(self):
        """No structured error JSON → _recover_from_error returns None."""
        runner = make_claude_runner()
        result = runner._recover_from_error("not json", "also not json", 1)
        assert result is None

    def test_valid_claude_json_on_error_returns_recovered_response(self):
        """Valid Claude JSON array in stdout on non-zero exit → recovered AgentResponse."""
        runner = make_claude_runner()
        result = runner._recover_from_error(CLAUDE_JSON_RESPONSE, "", 1)
        assert result is not None
        assert result.output == "test output"
        assert result.metadata.get("recovered_from_error") is True

    def test_malformed_stderr_falls_through(self):
        """Truncated/malformed JSON in stderr → _try_extract_error returns without raising."""
        stderr = '{"error": {"code": 429, "message": "truncated...'
        runner = make_claude_runner()
        result = runner._recover_from_error("", stderr, 1)
        assert result is None

    def test_error_json_in_stdout_fallback(self):
        """Empty stderr, error JSON in stdout → extracts and raises SubprocessError."""
        stdout = json.dumps({"error": {"code": 401, "message": "unauthorized"}})
        runner = make_claude_runner()
        with pytest.raises(SubprocessError):
            runner._try_extract_error(stdout, "", 1)


# ---------------------------------------------------------------------------
# Retryable errors
# ---------------------------------------------------------------------------


class TestClaudeRunnerRetryableErrors:
    """Test ClaudeRunner retryable error classification."""

    @pytest.mark.parametrize("code", [429, 503])
    def test_retryable_error_codes_raise_retryable_error(self, code: int):
        """Error codes 429 and 503 in stderr JSON → RetryableError."""
        stderr = json.dumps({"error": {"code": code, "message": "transient error"}})
        runner = make_claude_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error("", stderr, 1)

    @pytest.mark.parametrize("code", ["429", "503"])
    def test_retryable_string_error_codes_raise_retryable_error(self, code: str):
        """String error codes '429' and '503' → RetryableError (coerced to int)."""
        stderr = json.dumps({"error": {"code": code, "message": "transient error"}})
        runner = make_claude_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error("", stderr, 1)

    def test_non_retryable_error_code_raises_subprocess_error(self):
        """Non-retryable error code → SubprocessError (not RetryableError)."""
        stderr = json.dumps({"error": {"code": 401, "message": "unauthorized"}})
        runner = make_claude_runner()
        with pytest.raises(SubprocessError) as exc_info:
            runner._try_extract_error("", stderr, 1)
        assert not isinstance(exc_info.value, RetryableError)

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_integration_429_then_success(self, mock_exec):
        """Full run() with 429 on first call, success on second → await_count == 2."""
        error_stderr = json.dumps({"error": {"code": 429, "message": "rate limited"}})
        mock_exec.side_effect = [
            create_mock_process(stdout="", stderr=error_stderr, returncode=1),
            create_mock_process(stdout=CLAUDE_JSON_RESPONSE, returncode=0),
        ]
        runner = make_claude_runner()
        result = await runner.run(make_prompt_request(agent="claude", prompt="x", max_retries=2))
        assert result.output == "test output"
        assert mock_exec.await_count == 2

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_nonzero_returncode_raises_subprocess_error(self, mock_exec):
        """Non-zero exit with no recoverable content → SubprocessError raised by run()."""
        mock_exec.return_value = create_mock_process(
            stdout="", stderr="something went wrong", returncode=1
        )
        runner = make_claude_runner()
        with pytest.raises(SubprocessError):
            await runner.run(make_prompt_request(agent="claude", prompt="x"))

    def test_stderr_json_without_error_key_falls_through_to_stdout(self):
        """stderr JSON without 'error' key → stdout inspected; 429 in stdout → RetryableError."""
        stderr = json.dumps({"debug": "diagnostic info"})
        stdout = json.dumps({"error": {"code": 429, "message": "rate limited"}})
        runner = make_claude_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error(stdout, stderr, 1)

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_all_attempts_exhausted(self, mock_exec):
        """All 3 attempts return 429 → RetryableError raised; subprocess called 3 times."""
        error_stderr = json.dumps({"error": {"code": 429, "message": "rate limited"}})
        mock_exec.return_value = create_mock_process(stdout="", stderr=error_stderr, returncode=1)
        runner = make_claude_runner()
        request = make_prompt_request(agent="claude", prompt="x", max_retries=3)
        with pytest.raises(RetryableError):
            await runner.run(request)
        assert mock_exec.await_count == 3


# ---------------------------------------------------------------------------
# Error recovery (full run() with mocked subprocess)
# ---------------------------------------------------------------------------


class TestClaudeRunnerErrorRecovery:
    """Test ClaudeRunner error recovery from non-zero exit codes."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_recovers_from_partial_error(self, mock_exec):
        """Non-zero exit + valid JSON stdout → recovered response with metadata stamped."""
        mock_exec.return_value = create_mock_process(
            stdout=CLAUDE_JSON_RESPONSE,
            stderr="Warning: something minor",
            returncode=1,
        )
        runner = make_claude_runner()
        response = await runner.run(make_prompt_request(agent="claude", prompt="test"))
        assert response.output == "test output"
        assert response.metadata.get("recovered_from_error") is True
        assert response.metadata.get("original_exit_code") == 1
        assert response.metadata.get("stderr") == "Warning: something minor"

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_raises_when_recovery_impossible(self, mock_exec):
        """Non-zero exit + non-JSON stdout → SubprocessError with returncode and stderr."""
        mock_exec.return_value = create_mock_process(
            stdout="Fatal error: API key invalid",
            stderr="Error: 401 Unauthorized",
            returncode=1,
        )
        runner = make_claude_runner()
        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="claude", prompt="test"))
        assert exc_info.value.returncode == 1
        assert "Error: 401 Unauthorized" in exc_info.value.stderr

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_recovery_fails_when_is_error_true(self, mock_exec):
        """Non-zero exit + is_error=true stdout → SubprocessError (NOT ParseError).

        _recover_from_error catches ParseError from parse_output, then _try_extract_error
        finds no {"error": {...}} key in the result JSON → returns None → generic SubprocessError.
        """
        data = [{"type": "result", "is_error": True, "result": "Error: something failed"}]
        mock_exec.return_value = create_mock_process(
            stdout=json.dumps(data),
            stderr="",
            returncode=1,
        )
        runner = make_claude_runner()
        with pytest.raises(SubprocessError):
            await runner.run(make_prompt_request(agent="claude", prompt="x"))


# ---------------------------------------------------------------------------
# API error extraction (structured JSON in stderr/stdout)
# ---------------------------------------------------------------------------


class TestClaudeRunnerAPIErrorExtraction:
    """Test ClaudeRunner extracts structured API error details from error JSON."""

    @pytest.mark.parametrize(
        ("code", "message"),
        [
            (429, "rate limited"),
            (401, "unauthorized"),
        ],
        ids=["429-retryable", "401-auth"],
    )
    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_claude_error_json_surfaces_in_subprocess_error(self, mock_exec, code, message):
        """stderr error JSON → SubprocessError with 'Claude API error' and error code."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr=claude_error_json(code, message),
            returncode=1,
        )
        runner = make_claude_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="claude", prompt="x", max_retries=1))

        primary_message = exc_info.value.args[0]
        assert "Claude API error" in primary_message
        assert str(code) in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_structured_error_includes_command(self, mock_exec):
        """SubprocessError from structured error JSON includes the CLI command."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr=claude_error_json(401, "unauthorized"),
            returncode=1,
        )
        runner = make_claude_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="claude", prompt="x", max_retries=1))

        assert exc_info.value.command is not None
        assert "claude" in exc_info.value.command[0]

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_no_json_anywhere_falls_through_to_generic(self, mock_exec):
        """No JSON in stdout or stderr → generic SubprocessError without 'Claude API error'."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr="Connection refused: unable to reach API",
            returncode=1,
        )
        runner = make_claude_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="claude", prompt="x"))

        primary_message = exc_info.value.args[0]
        assert "Claude API error" not in primary_message
        assert "CLI command failed" in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_non_dict_json_stdout_does_not_crash(self, mock_exec):
        """Array of scalars in stdout → generic SubprocessError (no AttributeError)."""
        mock_exec.return_value = create_mock_process(
            stdout="[1, 2, 3]",
            stderr="something went wrong",
            returncode=1,
        )
        runner = make_claude_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="claude", prompt="x"))

        primary_message = exc_info.value.args[0]
        assert "Claude API error" not in primary_message

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_malformed_stderr_json_falls_through(self, mock_exec):
        """Truncated/malformed JSON in stderr → falls through to generic SubprocessError."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr='{"error": {"code": 429, "message": "truncated...',
            returncode=1,
        )
        runner = make_claude_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="claude", prompt="x"))

        primary_message = exc_info.value.args[0]
        assert "Claude API error" not in primary_message


# ---------------------------------------------------------------------------
# Dual-field recovery (valid stdout wins over error stderr)
# ---------------------------------------------------------------------------


class TestClaudeRunnerDualFieldRecovery:
    """Test _recover_from_error when stdout has valid result and stderr has error JSON."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_result_wins_over_error_json_on_nonzero_exit(self, mock_exec):
        """stdout=CLAUDE_JSON_RESPONSE, stderr=error JSON, returncode=1 → recovery wins.

        parse_output(stdout) succeeds first → _try_extract_error never called.
        Documents the cross-stream 'recovery wins' contract for Claude runner.
        """
        mock_exec.return_value = create_mock_process(
            stdout=CLAUDE_JSON_RESPONSE,
            stderr=claude_error_json(429, "rate limited"),
            returncode=1,
        )
        runner = make_claude_runner()
        request = make_prompt_request(agent="claude", prompt="x")

        response = await runner.run(request)

        assert response.output == "test output"
        assert response.metadata.get("recovered_from_error") is True


# ---------------------------------------------------------------------------
# Environment variable configuration
# ---------------------------------------------------------------------------


class TestClaudeRunnerEnvConfiguration:
    """Test ClaudeRunner environment variable configuration."""

    @patch.dict("os.environ", {"NEXUS_CLAUDE_PATH": "/opt/custom/claude"})
    def test_claude_runner_uses_custom_path_from_env(self):
        """ClaudeRunner uses NEXUS_CLAUDE_PATH if set."""
        runner = make_claude_runner()
        cmd = runner.build_command(make_prompt_request(agent="claude", prompt="test"))
        assert cmd[0] == "/opt/custom/claude"

    @patch.dict("os.environ", {"NEXUS_CLAUDE_MODEL": "claude-haiku-4-5-20251001"})
    def test_claude_runner_uses_default_model_from_env(self):
        """ClaudeRunner uses NEXUS_CLAUDE_MODEL as default if request.model is None."""
        runner = make_claude_runner()
        cmd = runner.build_command(make_prompt_request(agent="claude", prompt="test", model=None))
        assert "--model" in cmd
        assert "claude-haiku-4-5-20251001" in cmd

    @patch.dict("os.environ", {"NEXUS_CLAUDE_MODEL": "claude-haiku-4-5-20251001"})
    def test_claude_runner_request_model_overrides_env(self):
        """Request model overrides NEXUS_CLAUDE_MODEL env default."""
        runner = make_claude_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="claude", prompt="test", model="claude-sonnet-4-6")
        )
        assert "claude-sonnet-4-6" in cmd
        assert "claude-haiku-4-5-20251001" not in cmd


# ---------------------------------------------------------------------------
# Extract metadata
# ---------------------------------------------------------------------------


class TestClaudeRunnerExtractMetadata:
    """Test ClaudeRunner._extract_metadata static method."""

    @pytest.fixture
    def runner(self) -> ClaudeRunner:
        return make_claude_runner()

    def test_all_keys_present_extracted(self, runner):
        """All specified keys present in element are returned."""
        element = {
            "cost_usd": 0.005,
            "duration_ms": 1234,
            "num_turns": 2,
            "session_id": "sess-abc",
        }
        result = runner._extract_metadata(
            element, ("cost_usd", "duration_ms", "num_turns", "session_id")
        )
        assert result == {
            "cost_usd": 0.005,
            "duration_ms": 1234,
            "num_turns": 2,
            "session_id": "sess-abc",
        }

    def test_partial_keys_only_present_returned(self, runner):
        """Only keys present in element are returned; missing keys are omitted."""
        element = {"cost_usd": 0.001, "num_turns": 1}
        result = runner._extract_metadata(
            element, ("cost_usd", "duration_ms", "num_turns", "session_id")
        )
        assert result == {"cost_usd": 0.001, "num_turns": 1}

    def test_no_keys_present_returns_empty_dict(self, runner):
        """When none of the specified keys exist, empty dict is returned."""
        element = {"type": "result", "result": "text"}
        result = runner._extract_metadata(element, ("cost_usd", "duration_ms"))
        assert result == {}

    def test_empty_keys_tuple_returns_empty_dict(self, runner):
        """Empty keys tuple always produces empty dict."""
        element = {"cost_usd": 0.001, "duration_ms": 500}
        result = runner._extract_metadata(element, ())
        assert result == {}

    def test_does_not_mutate_source_element(self, runner):
        """Source element dict is not modified."""
        element = {"cost_usd": 0.001, "extra": "keep"}
        runner._extract_metadata(element, ("cost_usd",))
        assert "extra" in element


# ---------------------------------------------------------------------------
# Integration (full run() with mocked subprocess)
# ---------------------------------------------------------------------------


class TestClaudeRunnerIntegration:
    """Test ClaudeRunner.run() end-to-end with mocked subprocess."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_success_returns_parsed_response(self, mock_exec):
        """Successful run executes CLI with correct args and returns parsed response."""
        prompt = "test prompt"
        model = "claude-sonnet-4-6"
        mock_exec.return_value = create_mock_process(
            stdout=CLAUDE_JSON_RESPONSE,
            returncode=0,
        )
        runner = make_claude_runner()
        request = make_prompt_request(agent="claude", prompt=prompt, model=model)

        response = await runner.run(request)

        mock_exec.assert_awaited_once_with(
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            model,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert response.agent == "claude"
        assert response.output == "test output"
        assert "cost_usd" in response.metadata

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_subprocess_error_propagates(self, mock_exec):
        """Non-zero exit with no recoverable JSON → SubprocessError with returncode and stderr."""
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr="API key not found",
            returncode=1,
        )
        runner = make_claude_runner()

        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(make_prompt_request(agent="claude", prompt="x"))

        assert exc_info.value.returncode == 1
        assert "API key" in exc_info.value.stderr
