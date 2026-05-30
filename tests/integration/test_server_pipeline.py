# tests/integration/test_server_pipeline.py
"""Smoke tests for server-level behavior (no CLI binary required).

Real CLI pipelines are covered by runner-specific integration tests.
Progress reporting is covered by unit tests in tests/unit/test_server_progress.py.
"""

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.server import prompt


class TestServerInstructionsSmokeTest:
    """Smoke tests for server instructions (no CLI required)."""

    def test_instructions_mention_supported_runners(self) -> None:
        """Server instructions should mention supported production runners."""
        from nexus_mcp.server import mcp

        assert mcp.instructions is not None
        for runner in ("claude", "codex", "opencode"):
            assert runner in mcp.instructions


class TestServerPromptValidation:
    """Input-validation tests for prompt() that require no CLI."""

    async def test_prompt_rejects_unsupported_agent(self) -> None:
        """prompt() should raise ToolError for unknown agent names."""
        with pytest.raises(ToolError, match="nonexistent_agent_12345"):
            await prompt(
                cli="nonexistent_agent_12345",
                prompt="test",
            )
