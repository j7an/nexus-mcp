# tests/unit/runners/test_log_emitter.py
"""Tests for LogEmitter integration with runners.

Placed in tests/unit/runners/ so autouse fixtures (mock_cli_detection,
fast_retry_sleep) from conftest.py apply automatically.
"""

import json
from unittest.mock import patch as sync_patch

from nexus_mcp.exceptions import ParseError, RetryableError
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
        self.base_delay = 0.01
        self.max_delay = 0.01
        self.default_max_attempts = 1
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


class TestRetryEmit:
    """Retry loop emits warning on retryable errors."""

    @sync_patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_retry_emits_warning(self, mock_exec):
        """Retry loop emits warning before sleeping."""
        mock_exec.side_effect = [
            create_mock_process(stdout="", stderr="rate limited", returncode=1),
            create_mock_process(stdout=fake_json("test output")),
        ]
        runner = EmitterFakeRunner()
        calls: list[tuple[str, str]] = []

        async def collecting_emitter(level: str, message: str) -> None:
            calls.append((level, message))

        def patched_recover(stdout, stderr, returncode, command=None):
            raise RetryableError("429 rate limited", stderr=stderr, returncode=returncode)

        runner._recover_from_error = patched_recover

        await runner.run(make_prompt_request(max_retries=2), emitter=collecting_emitter)

        warning_calls = [(lvl, msg) for lvl, msg in calls if lvl == "warning"]
        assert any("Retryable error (attempt 1/2)" in msg for _, msg in warning_calls)


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
