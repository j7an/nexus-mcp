# tests/unit/runners/test_codex.py
"""Tests for CodexRunner.

Tests verify:
- CLINotFoundError when codex binary is absent
- build_command() constructs correct argument list
- build_command() respects model override and env default
- build_command() uses custom CLI path
- build_command() appends file_refs to prompt
- execution mode flags: sandbox, yolo, default
- parse_output() success and failure paths
- error handling: _recover_from_error, _try_extract_error
"""

import json
from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.exceptions import CLINotFoundError, ParseError, RetryableError, SubprocessError
from nexus_mcp.runners.codex import CodexRunner
from tests.fixtures import CODEX_NDJSON_RESPONSE, make_prompt_request

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_codex_runner() -> CodexRunner:
    """Create a CodexRunner using the autouse cli_detection_mocks fixture."""
    return CodexRunner()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestCodexRunnerInit:
    """Test CodexRunner initialisation and CLI detection."""

    def test_cli_not_found_raises_error(self):
        """CLINotFoundError raised when codex binary is absent."""
        not_found = CLIInfo(found=False, path=None)
        with (
            patch("nexus_mcp.runners.base.detect_cli", return_value=not_found),
            pytest.raises(CLINotFoundError) as exc_info,
        ):
            CodexRunner()
        assert exc_info.value.cli_name == "codex"

    def test_default_cli_path(self):
        """Default cli_path is 'codex' when NEXUS_CODEX_PATH is not set."""
        runner = make_codex_runner()
        assert runner.cli_path == "codex"

    def test_custom_cli_path_from_env(self):
        """cli_path is taken from NEXUS_CODEX_PATH env var."""
        with patch(
            "nexus_mcp.runners.base.get_agent_env",
            side_effect=lambda _agent, key, **_kw: "/custom/codex" if key == "PATH" else None,
        ):
            runner = make_codex_runner()
        assert runner.cli_path == "/custom/codex"


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------


class TestBuildCommand:
    """Test CodexRunner.build_command() argument construction."""

    def test_default_command_shape(self):
        """Default command: [cli_path, 'exec', prompt, '--json']."""
        runner = make_codex_runner()
        cmd = runner.build_command(make_prompt_request(agent="codex", prompt="hello"))
        assert cmd == ["codex", "exec", "hello", "--json"]

    def test_model_override(self):
        """request.model is forwarded as --model flag (adjacent to value)."""
        runner = make_codex_runner()
        cmd = runner.build_command(make_prompt_request(agent="codex", prompt="hi", model="o3"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "o3"

    def test_env_model_default(self):
        """default_model from env is used when request.model is None."""
        runner = make_codex_runner()
        runner.default_model = "o1-mini"
        cmd = runner.build_command(make_prompt_request(agent="codex", prompt="hi"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "o1-mini"

    def test_request_model_overrides_env_default(self):
        """request.model takes precedence over default_model."""
        runner = make_codex_runner()
        runner.default_model = "o1-mini"
        cmd = runner.build_command(make_prompt_request(agent="codex", prompt="hi", model="o3"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "o3"
        assert "o1-mini" not in cmd

    def test_file_refs_appended_to_prompt(self):
        """File references are appended to the prompt string."""
        runner = make_codex_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="codex", prompt="analyse", file_refs=["src/main.py"])
        )
        assert cmd[2].startswith("analyse")
        assert "src/main.py" in cmd[2]

    def test_sandbox_mode_adds_flags(self):
        """execution_mode='sandbox' adds --sandbox workspace-write."""
        runner = make_codex_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="codex", prompt="x", execution_mode="sandbox")
        )
        assert "--sandbox" in cmd
        assert "workspace-write" in cmd

    def test_yolo_mode_adds_flag(self):
        """execution_mode='yolo' adds --yolo flag."""
        runner = make_codex_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="codex", prompt="x", execution_mode="yolo")
        )
        assert "--yolo" in cmd

    def test_default_mode_no_extra_flags(self):
        """execution_mode='default' adds no extra approve flags."""
        runner = make_codex_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="codex", prompt="x", execution_mode="default")
        )
        assert "--sandbox" not in cmd
        assert "--yolo" not in cmd

    def test_build_command_model_with_yolo(self):
        """model + yolo: --model appears before --yolo in the command."""
        runner = make_codex_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="codex", prompt="x", model="o3", execution_mode="yolo")
        )
        assert "--model" in cmd
        assert "o3" in cmd
        assert "--yolo" in cmd
        assert cmd.index("--model") < cmd.index("--yolo")

    def test_build_command_model_with_sandbox(self):
        """model + sandbox: --model appears before --sandbox in the command."""
        runner = make_codex_runner()
        cmd = runner.build_command(
            make_prompt_request(agent="codex", prompt="x", model="o3", execution_mode="sandbox")
        )
        assert "--model" in cmd
        assert "o3" in cmd
        assert "--sandbox" in cmd
        assert "workspace-write" in cmd
        assert cmd.index("--model") < cmd.index("--sandbox")


# ---------------------------------------------------------------------------
# parse_output
# ---------------------------------------------------------------------------


class TestParseOutput:
    """Test CodexRunner.parse_output()."""

    def test_success_single_message(self):
        """Valid NDJSON with single agent_message → AgentResponse."""
        runner = make_codex_runner()
        result = runner.parse_output(CODEX_NDJSON_RESPONSE, "")
        assert result.agent == "codex"
        assert result.output == "pong"
        assert result.raw_output == CODEX_NDJSON_RESPONSE

    def test_multiple_items_joined(self):
        """Multiple agent_message events are joined with \\n\\n."""

        def event(text: str) -> str:
            return json.dumps(
                {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
            )

        ndjson = "\n".join([event("part1"), event("part2")])
        runner = make_codex_runner()
        result = runner.parse_output(ndjson, "")
        assert result.output == "part1\n\npart2"

    def test_empty_stdout_raises_parse_error(self):
        """Empty stdout → ParseError (no agent_message events found)."""
        runner = make_codex_runner()
        with pytest.raises(ParseError):
            runner.parse_output("", "")

    def test_no_agent_message_raises_parse_error(self):
        """NDJSON with no agent_message events → ParseError."""
        ndjson = '{"type": "thread.started"}\n{"type": "turn.completed"}'
        runner = make_codex_runner()
        with pytest.raises(ParseError):
            runner.parse_output(ndjson, "")

    def test_empty_text_field_falls_back_to_content_blocks(self):
        """agent_message with empty text field falls back to content block list."""
        event = json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "content": [{"text": "from content block"}],
                },
            }
        )
        runner = make_codex_runner()
        result = runner.parse_output(event, "")
        assert result.output == "from content block"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test CodexRunner._recover_from_error and _try_extract_error."""

    def test_no_json_in_stderr_or_stdout_returns_none(self):
        """No structured error JSON → _recover_from_error returns None."""
        runner = make_codex_runner()
        result = runner._recover_from_error("not json", "also not json", 1)
        assert result is None

    def test_valid_ndjson_on_error_returns_recovered_response(self):
        """Valid NDJSON in stdout on non-zero exit → recovered AgentResponse."""
        runner = make_codex_runner()
        result = runner._recover_from_error(CODEX_NDJSON_RESPONSE, "", 1)
        assert result is not None
        assert result.output == "pong"
        assert result.metadata.get("recovered_from_error") is True

    @pytest.mark.parametrize("code", [429, 503])
    def test_retryable_error_codes_raise_retryable_error(self, code: int):
        """Error codes 429 and 503 in stderr JSON → RetryableError."""
        stderr = json.dumps({"error": {"code": code, "message": "transient error"}})
        runner = make_codex_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error("", stderr, 1)

    @pytest.mark.parametrize("code", ["429", "503"])
    def test_retryable_string_error_codes_raise_retryable_error(self, code: str):
        """String error codes '429' and '503' in stderr JSON → RetryableError (coerced)."""
        stderr = json.dumps({"error": {"code": code, "message": "transient error"}})
        runner = make_codex_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error("", stderr, 1)

    def test_non_retryable_error_code_raises_subprocess_error(self):
        """Non-retryable error code in stderr JSON → SubprocessError (not RetryableError)."""
        stderr = json.dumps({"error": {"code": 401, "message": "unauthorized"}})
        runner = make_codex_runner()
        with pytest.raises(SubprocessError) as exc_info:
            runner._try_extract_error("", stderr, 1)
        assert not isinstance(exc_info.value, RetryableError)

    def test_malformed_json_in_stderr_returns_none(self):
        """Truncated/malformed JSON in stderr → _try_extract_error returns without raising."""
        stderr = '{"error": {"code": 429, "message": "truncated...'
        runner = make_codex_runner()
        # Should not raise — malformed JSON falls through silently
        result = runner._recover_from_error("", stderr, 1)
        assert result is None

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_nonzero_returncode_raises_subprocess_error(self, mock_exec):
        """Non-zero exit with no recoverable content → SubprocessError raised by run()."""
        from tests.fixtures import create_mock_process

        mock_exec.return_value = create_mock_process(
            stdout="", stderr="something went wrong", returncode=1
        )
        runner = make_codex_runner()
        with pytest.raises(SubprocessError):
            await runner.run(make_prompt_request(agent="codex", prompt="x"))

    def test_try_extract_error_stdout_fallback(self):
        """stderr="" + stdout has error JSON with code 503 → RetryableError raised."""
        stdout = json.dumps({"error": {"code": 503, "message": "Service unavailable"}})
        runner = make_codex_runner()
        with pytest.raises(RetryableError):
            runner._try_extract_error(stdout, "", 1)

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_on_503_full_loop(self, mock_exec):
        """503 on first attempt, success on second → called twice."""
        from tests.fixtures import create_mock_process

        error_stderr = json.dumps({"error": {"code": 503, "message": "Service unavailable"}})
        mock_exec.side_effect = [
            create_mock_process(stdout="", stderr=error_stderr, returncode=1),
            create_mock_process(stdout=CODEX_NDJSON_RESPONSE, returncode=0),
        ]
        runner = make_codex_runner()
        result = await runner.run(make_prompt_request(agent="codex", prompt="test", max_retries=2))
        assert result.output == "pong"
        assert mock_exec.await_count == 2

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_exhausted_all_503(self, mock_exec):
        """3x 503 → RetryableError raised, called 3 times."""
        from tests.fixtures import create_mock_process

        error_stderr = json.dumps({"error": {"code": 503, "message": "Service unavailable"}})
        mock_exec.return_value = create_mock_process(stdout="", stderr=error_stderr, returncode=1)
        runner = make_codex_runner()
        request = make_prompt_request(agent="codex", prompt="test", max_retries=3)
        with pytest.raises(RetryableError):
            await runner.run(request)
        assert mock_exec.await_count == 3

    def test_malformed_ndjson_lines_skipped(self):
        """Mixed valid/malformed lines → valid parts extracted, malformed skipped."""
        ndjson = "\n".join(
            [
                "not valid json {",
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "hello"},
                    }
                ),
                '{"type": "turn.completed"}',
            ]
        )
        runner = make_codex_runner()
        result = runner.parse_output(ndjson, "")
        assert result.output == "hello"

    def test_blank_lines_tolerated(self):
        """Blank lines between events → parsed correctly, blank lines skipped."""
        ndjson = "\n".join(
            [
                "",
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "world"},
                    }
                ),
                "",
                '{"type": "turn.completed"}',
            ]
        )
        runner = make_codex_runner()
        result = runner.parse_output(ndjson, "")
        assert result.output == "world"
