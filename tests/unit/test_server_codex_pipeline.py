# tests/unit/test_server_codex_pipeline.py
"""Pipeline integration tests: server tools → RunnerFactory → CodexRunner → subprocess.

Mocks only at asyncio.create_subprocess_exec — all layers above run for real:
    prompt()/batch_prompt() → RunnerFactory → CodexRunner → build_command
        → run_subprocess → [MOCK] → parse_output
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.server import prompt
from tests.fixtures import CODEX_NDJSON_RESPONSE, create_mock_process


def _codex_ndjson(text: str) -> str:
    """Build a minimal Codex NDJSON response with one agent_message event."""
    return json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": text},
        }
    )


# ---------------------------------------------------------------------------
# prompt() → CodexRunner pipeline
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_cli_detection")
class TestPromptCodexPipeline:
    """Full prompt() → CodexRunner pipeline tests.

    Mocks only asyncio.create_subprocess_exec. Everything above runs for real:
        prompt() → RunnerFactory.create("codex") → CodexRunner
            → build_command() → run_subprocess() → [MOCK] → parse_output()
    """

    async def test_success_returns_parsed_output(self, mock_subprocess):
        """Full success path: subprocess returns valid NDJSON → output text returned."""
        mock_subprocess.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE)

        result = await prompt("codex", "ping")

        assert result == "pong"
        assert mock_subprocess.call_count == 1
        args = list(mock_subprocess.call_args.args)
        assert "exec" in args
        assert "--json" in args

    async def test_model_reaches_subprocess_args(self, mock_subprocess):
        """model parameter is forwarded as --model flag in the subprocess command."""
        mock_subprocess.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE)

        await prompt("codex", "test", model="o3")

        args = list(mock_subprocess.call_args.args)
        assert "--model" in args
        assert "o3" in args

    async def test_sandbox_mode_reaches_subprocess_args(self, mock_subprocess):
        """execution_mode='sandbox' adds --sandbox workspace-write to subprocess args."""
        mock_subprocess.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE)

        await prompt("codex", "test", execution_mode="sandbox")

        args = list(mock_subprocess.call_args.args)
        assert "--sandbox" in args
        assert "workspace-write" in args
        assert "--yolo" not in args

    async def test_yolo_mode_reaches_subprocess_args(self, mock_subprocess):
        """execution_mode='yolo' adds --yolo to subprocess args."""
        mock_subprocess.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE)

        await prompt("codex", "test", execution_mode="yolo")

        args = list(mock_subprocess.call_args.args)
        assert "--yolo" in args

    async def test_empty_stdout_raises_tool_error(self, mock_subprocess):
        """Empty stdout (no agent_message events) → ParseError → ToolError."""
        mock_subprocess.return_value = create_mock_process(stdout="", returncode=0)

        with pytest.raises(ToolError, match=r"\[ParseError\]"):
            await prompt("codex", "test")

    async def test_recovery_from_nonzero_exit_with_valid_ndjson(self, mock_subprocess):
        """Non-zero exit with valid NDJSON → recovery path → output returned."""
        mock_subprocess.return_value = create_mock_process(
            stdout=CODEX_NDJSON_RESPONSE, returncode=1
        )

        result = await prompt("codex", "test")

        assert result == "pong"

    async def test_429_retries_then_succeeds(self, mock_subprocess, fast_retry_sleep):
        """HTTP 429 in stderr triggers retry; second attempt succeeds."""
        error_stderr = json.dumps({"error": {"code": 429, "message": "rate limited"}})
        mock_subprocess.side_effect = [
            create_mock_process(stdout="", stderr=error_stderr, returncode=1),
            create_mock_process(stdout=CODEX_NDJSON_RESPONSE),
        ]

        result = await prompt("codex", "test", max_retries=2)

        assert result == "pong"
        assert mock_subprocess.call_count == 2

    async def test_multiple_agent_messages_joined(self, mock_subprocess):
        """Multiple agent_message events are joined with \\n\\n in the output."""
        ndjson = "\n".join(
            [
                _codex_ndjson("part1"),
                _codex_ndjson("part2"),
            ]
        )
        mock_subprocess.return_value = create_mock_process(stdout=ndjson)

        result = await prompt("codex", "test")

        assert result == "part1\n\npart2"
