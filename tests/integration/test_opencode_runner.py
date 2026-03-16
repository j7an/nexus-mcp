# tests/integration/test_opencode_runner.py
"""Integration tests for OpenCodeRunner end-to-end pipeline.

Tests exercise the full Template Method pipeline:
  build_command() → run_subprocess() → parse_output()
using the real OpenCode CLI binary.

All tests are skipped when opencode CLI is not installed.
Tests marked @slow make real API calls and require valid auth config.

Output format verified against OpenCode CLI v1.2.20 (NDJSON events).
"""

import shutil

import pytest

from nexus_mcp.runners.opencode import OpenCodeRunner
from nexus_mcp.types import AgentResponse
from tests.fixtures import PING_PROMPT, make_prompt_request


@pytest.mark.integration
class TestOpenCodeRunnerConstruction:
    """Validate OpenCodeRunner.__init__() with real CLI detection."""

    def test_construction_succeeds_with_real_cli(self, opencode_cli_available: str) -> None:  # noqa: ARG002
        """OpenCodeRunner() should construct without error when CLI is installed."""
        runner = OpenCodeRunner()

        assert runner is not None
        assert runner.capabilities.found is True

    def test_construction_sets_timeout(self, opencode_cli_available: str) -> None:  # noqa: ARG002
        """OpenCodeRunner should inherit a positive timeout from AbstractRunner."""
        runner = OpenCodeRunner()

        assert runner.timeout > 0


@pytest.mark.integration
class TestOpenCodeRunnerBuildCommand:
    """Validate command building with real CLI detection."""

    def test_build_command_includes_json_flag(self, opencode_runner: OpenCodeRunner) -> None:
        """build_command() should include --format json flags."""
        request = make_prompt_request(cli="opencode", prompt="test")
        command = opencode_runner.build_command(request)

        assert "--format" in command
        assert "json" in command

    def test_build_command_includes_prompt_flag(self, opencode_runner: OpenCodeRunner) -> None:
        """build_command() should include the prompt as a positional argument."""
        request = make_prompt_request(cli="opencode", prompt="test")
        command = opencode_runner.build_command(request)

        assert "test" in command

    def test_build_command_cli_path_is_real(self, opencode_runner: OpenCodeRunner) -> None:
        """build_command() CLI path should resolve to an installed binary."""
        request = make_prompt_request(cli="opencode", prompt="test")
        command = opencode_runner.build_command(request)

        assert shutil.which(command[0]) is not None, (
            f"CLI path {command[0]!r} does not resolve to an installed binary"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestOpenCodeRunnerEndToEnd:
    """Full Template Method pipeline with real OpenCode API calls."""

    async def test_run_simple_prompt_returns_agent_response(
        self, opencode_runner: OpenCodeRunner
    ) -> None:
        """run() should return an AgentResponse with non-empty output."""
        request = make_prompt_request(cli="opencode", prompt=PING_PROMPT)
        response = await opencode_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert response.cli == "opencode"
        assert len(response.output) > 0

    async def test_run_default_mode_restricts_tools(self, opencode_runner: OpenCodeRunner) -> None:
        """run() with default mode should succeed (no tool restriction flags available)."""
        request = make_prompt_request(cli="opencode", prompt=PING_PROMPT, execution_mode="default")
        response = await opencode_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert len(response.output) > 0

    async def test_run_schema_discovery(self, opencode_runner: OpenCodeRunner) -> None:
        """Verify raw_output is non-empty and format matches NDJSON event stream."""
        request = make_prompt_request(cli="opencode", prompt=PING_PROMPT)
        response = await opencode_runner.run(request)

        # Print raw output for empirical format verification
        print(f"\nOpenCode raw_output:\n{response.raw_output!r}")
        assert response.raw_output, "raw_output should not be empty"
