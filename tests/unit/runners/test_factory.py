# tests/unit/runners/test_factory.py
"""Tests for RunnerFactory.

Tests verify:
- create("gemini") → GeminiRunner instance
- create("unknown") → UnsupportedAgentError
- list_agents() → ["gemini"]
"""

import pytest

from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.runners.gemini import GeminiRunner


class TestRunnerFactory:
    """Test RunnerFactory agent creation and listing."""

    def test_create_gemini_runner(self):
        """create("gemini") should return GeminiRunner instance."""
        factory = RunnerFactory()

        runner = factory.create("gemini")

        assert isinstance(runner, GeminiRunner)

    def test_create_unknown_agent_raises_error(self):
        """create("unknown") should raise UnsupportedAgentError."""
        factory = RunnerFactory()

        with pytest.raises(UnsupportedAgentError) as exc_info:
            factory.create("unknown-agent")

        assert exc_info.value.agent == "unknown-agent"
        assert "Unsupported agent: unknown-agent" in str(exc_info.value)

    def test_create_case_sensitive(self):
        """Agent names are case-sensitive: "Gemini" != "gemini"."""
        factory = RunnerFactory()

        with pytest.raises(UnsupportedAgentError):
            factory.create("Gemini")  # Capital G

    def test_list_agents_returns_supported_agents(self):
        """list_agents() should return list of supported agent names."""
        factory = RunnerFactory()

        agents = factory.list_agents()

        assert agents == ["gemini"]

    def test_create_returns_new_instance_each_time(self):
        """create() should return a new instance on each call (no singleton)."""
        factory = RunnerFactory()

        runner1 = factory.create("gemini")
        runner2 = factory.create("gemini")

        assert runner1 is not runner2  # Different instances
        assert type(runner1) is type(runner2)  # Same type
