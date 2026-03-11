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

import json
from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.exceptions import CLINotFoundError, ParseError, RetryableError, SubprocessError
from nexus_mcp.runners.claude import ClaudeRunner
from tests.fixtures import (
    CLAUDE_JSON_RESPONSE,
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
            side_effect=lambda agent, key, **kw: "/custom/claude" if key == "PATH" else None,
        ):
            runner = make_claude_runner()
        assert runner.cli_path == "/custom/claude"


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------


class TestBuildCommand:
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

    def test_file_refs_appended_to_prompt(self):
        """File references are appended to the prompt string."""
        runner = make_claude_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="claude", prompt="analyse", file_refs=["src/main.py"])
        )
        assert cmd[2].startswith("analyse")
        assert "src/main.py" in cmd[2]

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
# parse_output
# ---------------------------------------------------------------------------


class TestParseOutput:
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


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
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


class TestRetryableErrors:
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
        result = await runner.run(make_prompt_request(agent="claude", prompt="x"))
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
