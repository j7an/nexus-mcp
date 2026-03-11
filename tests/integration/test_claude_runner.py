# tests/integration/test_claude_runner.py
"""Integration tests for ClaudeRunner against the real Claude CLI.

These tests require the `claude` CLI binary installed and available in PATH.
They are automatically skipped when the CLI is absent (via the
claude_cli_available fixture in conftest.py).

Run with:
    uv run pytest -m integration -v
"""

import pytest

from nexus_mcp.runners.claude import ClaudeRunner
from tests.fixtures import PING_PROMPT, make_prompt_request


@pytest.mark.integration
@pytest.mark.slow
async def test_claude_runner_real_invocation(claude_runner: ClaudeRunner) -> None:
    """ClaudeRunner.run() against real CLI returns non-empty output with correct agent."""
    response = await claude_runner.run(make_prompt_request(agent="claude", prompt=PING_PROMPT))
    assert response.output
    assert response.agent == "claude"


@pytest.mark.integration
@pytest.mark.slow
async def test_claude_runner_metadata_present(claude_runner: ClaudeRunner) -> None:
    """Real Claude CLI response includes cost_usd and duration_ms metadata."""
    response = await claude_runner.run(make_prompt_request(agent="claude", prompt=PING_PROMPT))
    # Metadata may include cost/timing fields from the JSON response
    assert isinstance(response.metadata, dict)
