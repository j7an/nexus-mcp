# tests/integration/test_claude_runner.py
"""Integration tests for ClaudeRunner end-to-end pipeline.

Tests exercise the full Template Method pipeline:
  build_command() → run_subprocess() → parse_output()
using the real Claude Code CLI binary.

Tests marked @slow make real API calls and require valid credentials.

Run with:
    uv run pytest -m integration -v
"""

import shutil

import pytest

from nexus_mcp.runners.claude import ClaudeRunner
from nexus_mcp.types import AgentResponse
from tests.fixtures import PING_PROMPT, make_prompt_request


@pytest.mark.integration
class TestClaudeRunnerConstruction:
    """Validate ClaudeRunner.__init__() with real CLI detection."""

    def test_construction_succeeds_with_real_cli(self, claude_cli_available: str) -> None:  # noqa: ARG002
        """ClaudeRunner() should construct without error when CLI is installed."""
        runner = ClaudeRunner()

        assert runner is not None
        assert runner.capabilities.found is True

    def test_construction_sets_timeout(self, claude_cli_available: str) -> None:  # noqa: ARG002
        """ClaudeRunner should inherit a positive timeout from AbstractRunner."""
        runner = ClaudeRunner()

        assert runner.timeout > 0


@pytest.mark.integration
class TestClaudeRunnerBuildCommand:
    """Validate command building with real CLI detection."""

    def test_build_command_includes_json_flag(self, claude_runner: ClaudeRunner) -> None:
        """build_command() should include --output-format json flags."""
        request = make_prompt_request(agent="claude", prompt="test")
        command = claude_runner.build_command(request)

        assert "--output-format" in command
        assert "json" in command

    def test_build_command_includes_prompt_flag(self, claude_runner: ClaudeRunner) -> None:
        """build_command() should pass the prompt via the -p flag."""
        request = make_prompt_request(agent="claude", prompt="test")
        command = claude_runner.build_command(request)

        assert "-p" in command

    def test_build_command_cli_path_is_real(self, claude_runner: ClaudeRunner) -> None:
        """build_command() CLI path should resolve to an installed binary."""
        request = make_prompt_request(agent="claude", prompt="test")
        command = claude_runner.build_command(request)

        assert shutil.which(command[0]) is not None, (
            f"CLI path {command[0]!r} does not resolve to an installed binary"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestClaudeRunnerEndToEnd:
    """Full Template Method pipeline with real Claude API calls."""

    async def test_run_simple_prompt_returns_agent_response(
        self, claude_runner: ClaudeRunner
    ) -> None:
        """run() should return an AgentResponse with non-empty output."""
        request = make_prompt_request(agent="claude", prompt=PING_PROMPT)
        response = await claude_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert response.agent == "claude"
        assert len(response.output) > 0

    async def test_run_returns_cost_and_timing_metadata(self, claude_runner: ClaudeRunner) -> None:
        """Claude CLI JSON response includes cost_usd and duration_ms in metadata."""
        request = make_prompt_request(agent="claude", prompt=PING_PROMPT)
        response = await claude_runner.run(request)

        assert isinstance(response.metadata, dict)
        # Claude CLI always emits cost and timing in the result object
        assert "cost_usd" in response.metadata
        assert "duration_ms" in response.metadata

    async def test_run_with_model_flag(self, claude_runner: ClaudeRunner) -> None:
        """run() with an explicit model should succeed and return output."""
        request = make_prompt_request(
            agent="claude", prompt=PING_PROMPT, model="claude-haiku-4-5-20251001"
        )
        response = await claude_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert len(response.output) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestClaudeRunnerErrorPath:
    """Validate error propagation when the real CLI fails.

    Uses an invalid model name to reliably trigger a non-zero exit from the CLI.
    This exercises the full error path:
      server.py → base.py run() → _recover_from_error() or SubprocessError

    Flakiness risk: CLI error messages and exit codes may vary across versions.
    The assertion is intentionally broad (SubprocessError OR recovered_from_error)
    to remain valid even if recovery behavior changes.
    """

    async def test_run_with_invalid_model_raises_or_recovers(
        self, claude_runner: ClaudeRunner
    ) -> None:
        """run() with a nonexistent model should either raise SubprocessError or recover."""
        from nexus_mcp.exceptions import SubprocessError

        request = make_prompt_request(
            agent="claude", prompt="ping", model="nonexistent-model-xyz-99"
        )
        try:
            response = await claude_runner.run(request)
            # Recovery path: error was caught, metadata records it
            assert response.metadata.get("recovered_from_error") is True, (
                "Expected SubprocessError or recovered_from_error=True for invalid model, "
                f"got response: {response.output!r}"
            )
        except SubprocessError:
            pass  # Error propagation path — also correct
