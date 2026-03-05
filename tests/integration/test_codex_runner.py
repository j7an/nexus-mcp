# tests/integration/test_codex_runner.py
"""Integration tests for CodexRunner end-to-end pipeline.

Tests exercise the full Template Method pipeline:
  build_command() → run_subprocess() → parse_output()
using the real Codex CLI binary.

Tests marked @slow make real API calls and require a valid API key / auth config.
"""

import shutil

import pytest

from nexus_mcp.runners.codex import CodexRunner
from nexus_mcp.types import AgentResponse
from tests.fixtures import PING_PROMPT, make_prompt_request


@pytest.mark.integration
class TestCodexRunnerConstruction:
    """Validate CodexRunner.__init__() with real CLI detection."""

    def test_construction_succeeds_with_real_cli(self, codex_cli_available: str) -> None:  # noqa: ARG002
        """CodexRunner() should construct without error when CLI is installed."""
        runner = CodexRunner()

        assert runner is not None
        assert runner.capabilities.found is True

    def test_construction_sets_timeout(self, codex_cli_available: str) -> None:  # noqa: ARG002
        """CodexRunner should inherit a positive timeout from AbstractRunner."""
        runner = CodexRunner()

        assert runner.timeout > 0


@pytest.mark.integration
class TestCodexRunnerBuildCommand:
    """Validate command building with real CLI detection."""

    def test_build_command_includes_json_flag(self, codex_runner: CodexRunner) -> None:
        """build_command() should include --json flag."""
        request = make_prompt_request(agent="codex", prompt="test")
        command = codex_runner.build_command(request)

        assert "--json" in command

    def test_build_command_includes_exec_subcommand(self, codex_runner: CodexRunner) -> None:
        """build_command() should use 'exec' subcommand."""
        request = make_prompt_request(agent="codex", prompt="test")
        command = codex_runner.build_command(request)

        assert "exec" in command

    def test_build_command_cli_path_is_real(self, codex_runner: CodexRunner) -> None:
        """build_command() CLI path should resolve to an installed binary."""
        request = make_prompt_request(agent="codex", prompt="test")
        command = codex_runner.build_command(request)

        assert shutil.which(command[0]) is not None, (
            f"CLI path {command[0]!r} does not resolve to an installed binary"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestCodexRunnerEndToEnd:
    """Full Template Method pipeline with real Codex API calls."""

    async def test_run_simple_prompt_returns_agent_response(
        self, codex_runner: CodexRunner
    ) -> None:
        """run() should return an AgentResponse with non-empty output."""
        request = make_prompt_request(agent="codex", prompt=PING_PROMPT)
        response = await codex_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert response.agent == "codex"
        assert len(response.output) > 0

    async def test_run_returns_parsed_output(self, codex_runner: CodexRunner) -> None:
        """run() should populate all AgentResponse fields from parsed NDJSON."""
        request = make_prompt_request(agent="codex", prompt=PING_PROMPT)
        response = await codex_runner.run(request)

        assert isinstance(response.output, str)
        assert isinstance(response.raw_output, str)
        assert isinstance(response.metadata, dict)
        assert len(response.raw_output) > 0
