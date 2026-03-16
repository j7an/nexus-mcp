# tests/unit/test_server_claude_pipeline.py
"""Pipeline integration tests: server tools → RunnerFactory → ClaudeRunner → subprocess.

Mocks only at asyncio.create_subprocess_exec — all layers above run for real:
    prompt()/batch_prompt() → RunnerFactory → ClaudeRunner → build_command
        → run_subprocess → [MOCK] → parse_output

This validates the full request/response path that server tests (mock at RunnerFactory)
and runner tests (mock at subprocess) independently miss.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.server import batch_prompt, prompt
from tests.fixtures import (
    CLAUDE_NOISY_STDOUT,
    create_mock_process,
    make_agent_task,
)
from tests.fixtures import (
    claude_error_json as _claude_error_json,
)

# ---------------------------------------------------------------------------
# JSON response builders
# ---------------------------------------------------------------------------


def _claude_json(text: str) -> str:
    """Build a Claude Code CLI JSON response string (single-object format)."""
    return json.dumps(
        [
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": text,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Class 1: prompt() → ClaudeRunner pipeline
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_cli_detection")
class TestPromptClaudePipeline:
    """Tests for the full prompt() → ClaudeRunner pipeline.

    Mocks only asyncio.create_subprocess_exec. Everything above runs for real:
        prompt() → RunnerFactory.create("claude") → ClaudeRunner
            → build_command() → run_subprocess() → [MOCK] → parse_output()
    """

    async def test_success(self, mock_subprocess):
        """Full success path: subprocess returns valid JSON → prompt() returns output text."""
        mock_subprocess.return_value = create_mock_process(stdout=_claude_json("Hello from Claude"))

        result = await prompt("claude", "Say hello")

        assert result == "Hello from Claude"
        assert mock_subprocess.call_count == 1
        args = list(mock_subprocess.call_args.args)
        assert "-p" in args
        assert "--output-format" in args
        assert "json" in args

    async def test_model_forwarding(self, mock_subprocess):
        """model parameter is forwarded as --model flag in the subprocess command."""
        mock_subprocess.return_value = create_mock_process(stdout=_claude_json("ok"))

        await prompt("claude", "test", model="claude-sonnet-4-6")

        args = list(mock_subprocess.call_args.args)
        assert "--model" in args
        assert "claude-sonnet-4-6" in args

    async def test_yolo_mode_adds_flag(self, mock_subprocess):
        """execution_mode='yolo' adds --dangerously-skip-permissions flag."""
        mock_subprocess.return_value = create_mock_process(stdout=_claude_json("ok"))

        await prompt("claude", "test", execution_mode="yolo")

        args = list(mock_subprocess.call_args.args)
        assert "--dangerously-skip-permissions" in args

    async def test_parse_error_raises_tool_error(self, mock_subprocess):
        """Invalid JSON stdout → ParseError → ToolError with [ParseError] prefix."""
        mock_subprocess.return_value = create_mock_process(stdout="not valid json", returncode=0)

        with pytest.raises(ToolError, match=r"\[ParseError\]"):
            await prompt("claude", "test")

    async def test_401_raises_tool_error(self, mock_subprocess):
        """Non-retryable API error (401) → SubprocessError → ToolError with [SubprocessError]."""
        error_json = _claude_error_json(401, "Unauthorized: invalid API key")
        mock_subprocess.return_value = create_mock_process(stderr=error_json, returncode=1)

        with pytest.raises(ToolError, match=r"\[SubprocessError\]"):
            await prompt("claude", "test")

    async def test_429_retry_then_success(self, mock_subprocess, fast_retry_sleep):
        """HTTP 429 rate limit triggers retry; second attempt succeeds."""
        error_json = _claude_error_json(429, "Rate limited")
        mock_subprocess.side_effect = [
            create_mock_process(stderr=error_json, returncode=1),
            create_mock_process(stdout=_claude_json("ok after retry")),
        ]

        result = await prompt("claude", "test", max_retries=2)

        assert result == "ok after retry"
        assert mock_subprocess.call_count == 2

    async def test_recovery_from_nonzero_exit(self, mock_subprocess):
        """Non-zero exit with valid response JSON → recovery path → output returned."""
        mock_subprocess.return_value = create_mock_process(
            stdout=_claude_json("recovered output"), returncode=1
        )

        result = await prompt("claude", "test")

        assert result == "recovered output"

    async def test_noisy_stdout_parsed(self, mock_subprocess):
        """Log lines before JSON → still parsed correctly via fallback."""
        mock_subprocess.return_value = create_mock_process(stdout=CLAUDE_NOISY_STDOUT, returncode=0)

        result = await prompt("claude", "test")

        assert result == "test output"

    async def test_exhausted_retries_503(self, mock_subprocess, fast_retry_sleep):
        """HTTP 503 that always fails → exhausts all retries → ToolError with [RetryableError]."""
        error_json = _claude_error_json(503, "Service unavailable")
        mock_subprocess.return_value = create_mock_process(stderr=error_json, returncode=1)

        with pytest.raises(ToolError, match=r"\[RetryableError\]"):
            await prompt("claude", "test", max_retries=3)

        assert mock_subprocess.call_count == 3


# ---------------------------------------------------------------------------
# Class 2: batch_prompt() → ClaudeRunner pipeline
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_cli_detection")
class TestBatchPromptClaudePipeline:
    """Tests for the full batch_prompt() → ClaudeRunner pipeline."""

    async def test_all_succeed_with_labels(self, mock_subprocess):
        """Two tasks both succeed; auto-assigned labels include 'claude' and 'claude-2'."""
        mock_subprocess.return_value = create_mock_process(stdout=_claude_json("ok"))
        tasks = [
            make_agent_task(agent="claude", prompt="first"),
            make_agent_task(agent="claude", prompt="second"),
        ]

        response = await batch_prompt(tasks=tasks)

        assert response.succeeded == 2
        labels = {r.label for r in response.results}
        assert "claude" in labels
        assert "claude-2" in labels

    async def test_mixed_success_and_error(self, mock_subprocess):
        """One task returns valid JSON; one returns invalid JSON → succeeded=1, failed=1."""

        def side_effect(*args, **_kwargs):
            cmd = list(args)
            prompt_text = cmd[cmd.index("-p") + 1]
            if "succeed" in prompt_text:
                return create_mock_process(stdout=_claude_json("ok"))
            return create_mock_process(stdout="not valid json", returncode=0)

        mock_subprocess.side_effect = side_effect
        tasks = [
            make_agent_task(agent="claude", prompt="please succeed"),
            make_agent_task(agent="claude", prompt="will fail"),
        ]

        response = await batch_prompt(tasks=tasks)

        assert response.succeeded == 1
        assert response.failed == 1
        failed_result = next(r for r in response.results if r.error)
        assert failed_result.error_type == "ParseError"

    async def test_dict_task_deserialization(self, mock_subprocess):
        """Pass raw dicts as tasks; validates Docket JSON round-trip deserialization fix."""
        mock_subprocess.return_value = create_mock_process(stdout=_claude_json("ok"))
        # Pass dicts instead of AgentTask objects (simulates Docket serialization round-trip)
        tasks = [
            {"agent": "claude", "prompt": "task one", "execution_mode": "default"},
            {"agent": "claude", "prompt": "task two", "execution_mode": "default"},
        ]

        response = await batch_prompt(tasks=tasks)  # type: ignore[arg-type]

        assert response.succeeded == 2
