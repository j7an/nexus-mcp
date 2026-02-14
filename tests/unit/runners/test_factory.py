# tests/unit/runners/test_factory.py
"""Tests for RunnerFactory.

Tests verify:
- create("gemini") → GeminiRunner instance
- create("unknown") → UnsupportedAgentError
- list_agents() → ["gemini"]
"""

from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.runners.gemini import GeminiRunner


@pytest.fixture(autouse=True)
def mock_cli_detection():
    """Auto-mock CLI detection for factory tests.

    RunnerFactory.create("gemini") calls GeminiRunner() which needs
    detect_cli() and get_cli_version() mocked.
    """
    with (
        patch("nexus_mcp.runners.gemini.detect_cli") as mock_detect,
        patch("nexus_mcp.runners.gemini.get_cli_version", return_value="0.12.0"),
    ):
        mock_detect.return_value = CLIInfo(found=True, path="/usr/bin/gemini")
        yield mock_detect


class TestRunnerFactory:
    """Test RunnerFactory agent creation and listing."""

    def test_create_gemini_runner(self):
        """create("gemini") should return GeminiRunner instance."""
        runner = RunnerFactory.create("gemini")

        assert isinstance(runner, GeminiRunner)

    def test_create_unknown_agent_raises_error(self):
        """create("unknown") should raise UnsupportedAgentError."""
        with pytest.raises(UnsupportedAgentError) as exc_info:
            RunnerFactory.create("unknown-agent")

        assert exc_info.value.agent == "unknown-agent"
        assert "Unsupported agent: unknown-agent" in str(exc_info.value)

    def test_create_case_sensitive(self):
        """Agent names are case-sensitive: "Gemini" != "gemini"."""
        with pytest.raises(UnsupportedAgentError):
            RunnerFactory.create("Gemini")  # Capital G

    def test_list_agents_returns_supported_agents(self):
        """list_agents() should return list of supported agent names."""
        agents = RunnerFactory.list_agents()

        assert agents == ["gemini"]

    def test_create_returns_new_instance_each_time(self):
        """create() should return a new instance on each call (no singleton)."""
        runner1 = RunnerFactory.create("gemini")
        runner2 = RunnerFactory.create("gemini")

        assert runner1 is not runner2  # Different instances
        assert type(runner1) is type(runner2)  # Same type
