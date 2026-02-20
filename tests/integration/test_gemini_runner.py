# tests/integration/test_gemini_runner.py
"""Integration tests for GeminiRunner end-to-end pipeline.

Tests exercise the full Template Method pipeline:
  build_command() → run_subprocess() → parse_output()
using the real Gemini CLI binary.

Tests marked @slow make real Gemini API calls and require valid credentials.
"""

import shutil

import pytest

from nexus_mcp.runners.gemini import GeminiRunner
from nexus_mcp.types import AgentResponse
from tests.fixtures import PING_PROMPT, make_prompt_request


@pytest.mark.integration
class TestGeminiRunnerConstruction:
    """Validate GeminiRunner.__init__() with real CLI detection."""

    def test_construction_succeeds_with_real_cli(self, gemini_cli_available: str) -> None:  # noqa: ARG002
        """GeminiRunner() should construct without error when CLI is installed."""
        runner = GeminiRunner()

        assert runner is not None
        assert runner.capabilities.found is True

    def test_construction_sets_timeout(self, gemini_cli_available: str) -> None:  # noqa: ARG002
        """GeminiRunner should inherit a positive timeout from AbstractRunner."""
        runner = GeminiRunner()

        assert runner.timeout > 0


@pytest.mark.integration
@pytest.mark.slow
class TestGeminiRunnerEndToEnd:
    """Full Template Method pipeline with real Gemini API calls."""

    async def test_run_simple_prompt_returns_agent_response(
        self, gemini_runner: GeminiRunner
    ) -> None:
        """run() should return an AgentResponse with non-empty output."""
        request = make_prompt_request(prompt=PING_PROMPT)
        response = await gemini_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert response.agent == "gemini"
        assert len(response.output) > 0

    async def test_run_returns_json_parsed_output(self, gemini_runner: GeminiRunner) -> None:
        """run() should populate all AgentResponse fields from parsed JSON."""
        request = make_prompt_request(prompt=PING_PROMPT)
        response = await gemini_runner.run(request)

        assert isinstance(response.output, str)
        assert isinstance(response.raw_output, str)
        assert isinstance(response.metadata, dict)
        # raw_output should contain the original JSON from the CLI
        assert len(response.raw_output) > 0

    async def test_run_with_model_flag(self, gemini_runner: GeminiRunner) -> None:
        """run() with an explicit model should succeed and return output."""
        request = make_prompt_request(prompt=PING_PROMPT, model="gemini-2.5-flash")
        response = await gemini_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert len(response.output) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestGeminiRunnerErrorPath:
    """Validate error propagation when the real CLI fails.

    Uses an invalid model name to reliably trigger a non-zero exit from the CLI.
    This exercises the full error path:
      server.py:69 → base.py run() → _recover_from_error() or SubprocessError

    Flakiness risk: CLI error messages and exit codes may vary across versions.
    The assertion is intentionally broad (SubprocessError OR recovered_from_error)
    to remain valid even if recovery behavior changes.
    """

    async def test_run_with_invalid_model_raises_or_recovers(
        self, gemini_runner: GeminiRunner
    ) -> None:
        """run() with a nonexistent model should either raise SubprocessError or recover."""
        from nexus_mcp.exceptions import SubprocessError

        request = make_prompt_request(prompt="ping", model="nonexistent-model-xyz-99")
        try:
            response = await gemini_runner.run(request)
            # Recovery path: error was caught, metadata records it
            assert response.metadata.get("recovered_from_error") is True, (
                "Expected SubprocessError or recovered_from_error=True for invalid model, "
                f"got response: {response.output!r}"
            )
        except SubprocessError:
            pass  # Error propagation path — also correct


@pytest.mark.integration
class TestGeminiRunnerBuildCommand:
    """Validate command building uses real capability detection."""

    def test_build_command_includes_json_flag(self, gemini_runner: GeminiRunner) -> None:
        """build_command() should include --output-format json for modern CLI versions."""
        request = make_prompt_request(prompt="test")
        command = gemini_runner.build_command(request)

        assert "--output-format" in command
        assert "json" in command

    def test_build_command_cli_path_is_real(self, gemini_runner: GeminiRunner) -> None:
        """build_command() CLI path should resolve to an installed binary."""
        request = make_prompt_request(prompt="test")
        command = gemini_runner.build_command(request)

        # command[0] is the CLI path — verify it resolves to a real binary
        assert shutil.which(command[0]) is not None, (
            f"CLI path {command[0]!r} does not resolve to an installed binary"
        )
