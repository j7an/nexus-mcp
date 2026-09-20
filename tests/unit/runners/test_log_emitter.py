# tests/unit/runners/test_log_emitter.py
"""Tests for LogEmitter integration with runners.

Placed in tests/unit/runners/ so the autouse CLI-detection fixture from
conftest.py applies automatically.
"""

import json
from unittest.mock import patch as sync_patch

from nexus_mcp.exceptions import ParseError
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, PromptRequest
from tests.fixtures import (
    REPRESENTATIVE_CLI,
    create_mock_process,
    make_prompt_request,
)


def fake_json(output: str) -> str:
    return json.dumps({"response": output})


class EmitterFakeRunner(AbstractRunner):
    """Minimal runner that exercises AbstractRunner emitter paths."""

    AGENT_NAME = REPRESENTATIVE_CLI

    def __init__(self) -> None:
        self.timeout = 30
        self.output_limit = 50_000
        self.default_model = None
        self.cli_path = self.AGENT_NAME

    def build_command(self, request: PromptRequest) -> list[str]:
        return [self.cli_path, "-p", self._build_prompt(request)]

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        try:
            output = json.loads(stdout)["response"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise ParseError("Failed to parse fake output", raw_output=stdout) from exc
        return AgentResponse(cli=self.AGENT_NAME, output=output, raw_output=stdout)


class TestEmitterThreading:
    """Emitter is threaded from run() through _execute()."""

    @sync_patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_passes_emitter_to_execute(self, mock_exec):
        """When emitter is provided, _execute receives it and emits info on subprocess launch."""
        mock_exec.return_value = create_mock_process(stdout=fake_json("test output"))
        runner = EmitterFakeRunner()
        calls: list[tuple[str, str]] = []

        async def collecting_emitter(level: str, message: str) -> None:
            calls.append((level, message))

        await runner.run(make_prompt_request(), emitter=collecting_emitter)

        info_calls = [(lvl, msg) for lvl, msg in calls if lvl == "info"]
        assert len(info_calls) >= 1
        assert f"Running {REPRESENTATIVE_CLI}" in info_calls[0][1]

    @sync_patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_without_emitter_uses_default(self, mock_exec):
        """When no emitter is provided, _default_log_emitter is used (no crash)."""
        mock_exec.return_value = create_mock_process(stdout=fake_json("test output"))
        runner = EmitterFakeRunner()

        response = await runner.run(make_prompt_request())
        assert response.output == "test output"


class TestTruncationEmit:
    """Output truncation emits info with size details."""

    @sync_patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_truncation_emits_info(self, mock_exec):
        """Truncation emits info log with original and truncated sizes."""
        big_output = "x" * 60000  # > 50KB default limit
        mock_exec.return_value = create_mock_process(stdout=fake_json(big_output))
        runner = EmitterFakeRunner()
        calls: list[tuple[str, str]] = []

        async def collecting_emitter(level: str, message: str) -> None:
            calls.append((level, message))

        await runner.run(make_prompt_request(), emitter=collecting_emitter)

        info_calls = [(lvl, msg) for lvl, msg in calls if lvl == "info"]
        assert any("Output truncated" in msg for _, msg in info_calls)


class TestRecoveryEmit:
    """Error recovery emits warning."""

    @sync_patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_recovery_emits_warning(self, mock_exec):
        """Recovery from non-zero exit code emits warning."""
        mock_exec.return_value = create_mock_process(
            stdout=fake_json("test output"), stderr="some warning", returncode=1
        )
        runner = EmitterFakeRunner()
        calls: list[tuple[str, str]] = []

        async def collecting_emitter(level: str, message: str) -> None:
            calls.append((level, message))

        await runner.run(make_prompt_request(), emitter=collecting_emitter)

        warning_calls = [(lvl, msg) for lvl, msg in calls if lvl == "warning"]
        assert any(
            "Recovered response from non-zero exit code 1" in msg for _, msg in warning_calls
        )
