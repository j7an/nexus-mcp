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
    async def test_prompt_agent_returns_response(self, mock_factory, progress):
        """prompt_agent dispatches to runner and returns output text."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Agent response")
        mock_factory.create.return_value = mock_runner

        result = await prompt_agent(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
        )

        assert result == "Agent response"
        mock_factory.create.assert_called_once_with("gemini")
        call_args = mock_runner.run.call_args.args[0]
        assert call_args.prompt == "Test prompt"
        assert call_args.context == {}
        assert call_args.execution_mode == "default"
        assert call_args.model is None

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_passes_execution_mode(self, mock_factory, progress):
        """execution_mode is passed through to the PromptRequest."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Done")
        mock_factory.create.return_value = mock_runner

        await prompt_agent(
            agent="gemini",
            prompt="Complex task",
            progress=progress,
            execution_mode="yolo",
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.execution_mode == "yolo"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_passes_model(self, mock_factory, progress):
        """model parameter is passed through to the PromptRequest."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Done")
        mock_factory.create.return_value = mock_runner

        await prompt_agent(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
            model="gemini-2.5-flash",
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.model == "gemini-2.5-flash"

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_passes_context(self, mock_factory, progress):
        """context is passed through to the PromptRequest."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Done")
        mock_factory.create.return_value = mock_runner

        await prompt_agent(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
            context={"key": "value"},
        )

        call_args = mock_runner.run.call_args.args[0]
        assert call_args.context == {"key": "value"}

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_reports_progress(self, mock_factory, progress):
        """progress.set_total and progress.increment are called during execution."""
        mock_runner = AsyncMock()
        mock_runner.run.return_value = make_agent_response(output="Done")
        mock_factory.create.return_value = mock_runner

        await prompt_agent(
            agent="gemini",
            prompt="Test prompt",
            progress=progress,
        )

        progress.set_total.assert_called_once_with(100)
        assert progress.increment.call_count == 4
        assert sum(c.args[0] for c in progress.increment.call_args_list) == 100

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_handles_unsupported_agent(self, mock_factory, progress):
        """UnsupportedAgentError propagates when factory cannot create runner."""
        mock_factory.create.side_effect = UnsupportedAgentError("unknown_agent")

        with pytest.raises(UnsupportedAgentError, match="unknown_agent"):
            await prompt_agent(
                agent="unknown_agent",
                prompt="Test prompt",
                progress=progress,
            )

    @patch("nexus_mcp.server.RunnerFactory")
    async def test_prompt_agent_handles_subprocess_error(self, mock_factory, progress):
        """SubprocessError propagates when runner.run() fails."""
        mock_runner = AsyncMock()
        mock_runner.run.side_effect = SubprocessError(
            "CLI command failed", stderr="error output", returncode=1
        )
        mock_factory.create.return_value = mock_runner

        with pytest.raises(SubprocessError, match="CLI command failed"):
            await prompt_agent(
                agent="gemini",
                prompt="Test prompt",
                progress=progress,
            )


class TestListAgents:
    """Tests for the list_agents tool function."""

    def test_list_agents_returns_supported_agents(self):
        """list_agents returns exactly the supported agent names."""
        agents = list_agents()
        assert agents == ["gemini"]
