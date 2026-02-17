# tests/unit/test_server.py
"""Tests for the FastMCP server and tool functions.

Mocking strategy: Mock at the RunnerFactory/runner boundary, NOT subprocess level.
Server tests should be decoupled from runner internals.
"""

from unittest.mock import AsyncMock, patch

import pytest

from nexus_mcp.exceptions import SubprocessError, UnsupportedAgentError
from nexus_mcp.server import list_agents, prompt_agent
from tests.fixtures import make_agent_response


class TestPromptAgent:
    """Tests for the prompt_agent tool function."""

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_returns_response(self, mock_factory):
        """prompt_agent dispatches to runner and returns output text."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Agent response")
        mock_factory.create.return_value = mock_runner

        progress = AsyncMock()

        result = await prompt_agent(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
        )

        assert result == "Agent response"
        mock_factory.create.assert_called_once_with("gemini")

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_passes_execution_mode(self, mock_factory):
        """execution_mode is passed through to the PromptRequest."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Done")
        mock_factory.create.return_value = mock_runner

        progress = AsyncMock()

        await prompt_agent(
            agent="gemini",
            prompt="Complex task",
            progress=progress,
            execution_mode="yolo",
        )

        call_args = mock_runner.run.call_args[0][0]
        assert call_args.execution_mode == "yolo"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_passes_model(self, mock_factory):
        """model parameter is passed through to the PromptRequest."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Done")
        mock_factory.create.return_value = mock_runner

        progress = AsyncMock()

        await prompt_agent(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
            model="gemini-2.5-flash",
        )

        call_args = mock_runner.run.call_args[0][0]
        assert call_args.model == "gemini-2.5-flash"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_reports_progress(self, mock_factory):
        """progress.set_total and progress.increment are called during execution."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Done")
        mock_factory.create.return_value = mock_runner

        progress = AsyncMock()

        await prompt_agent(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
        )

        progress.set_total.assert_called_once_with(100)
        assert progress.increment.call_count >= 2

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_handles_unsupported_agent(self, mock_factory):
        """UnsupportedAgentError propagates when factory cannot create runner."""
        mock_factory.create.side_effect = UnsupportedAgentError("unknown_agent")

        progress = AsyncMock()

        with pytest.raises(UnsupportedAgentError, match="unknown_agent"):
            await prompt_agent(
                agent="unknown_agent",
                prompt="Test prompt",
                progress=progress,
            )

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_handles_subprocess_error(self, mock_factory):
        """SubprocessError propagates when runner.run() fails."""
        mock_runner = AsyncMock()
        mock_runner.run.side_effect = SubprocessError(
            "CLI command failed", stderr="error output", returncode=1
        )
        mock_factory.create.return_value = mock_runner

        progress = AsyncMock()

        with pytest.raises(SubprocessError, match="CLI command failed"):
            await prompt_agent(
                agent="gemini",
                prompt="Test prompt",
                progress=progress,
            )


class TestListAgents:
    """Tests for the list_agents tool function."""

    def test_list_agents_returns_supported_agents(self):
        """list_agents returns a list containing supported agent names."""
        agents = list_agents()
        assert isinstance(agents, list)
        assert "gemini" in agents
