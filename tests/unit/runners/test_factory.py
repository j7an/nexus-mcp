# tests/unit/runners/test_factory.py
"""Tests for RunnerFactory.

Tests verify:
- create("gemini") → GeminiRunner instance
- create("unknown") → UnsupportedAgentError
- list_agents() → ["gemini"]
- create() returns cached instance for same agent
- clear_cache() forces fresh instance on next create()
"""

import pytest

from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.runners.gemini import GeminiRunner


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

    def test_create_returns_cached_instance(self):
        """create() should return the same cached instance for the same agent."""
        runner1 = RunnerFactory.create("gemini")
        runner2 = RunnerFactory.create("gemini")

        assert runner1 is runner2  # Same cached instance

    def test_clear_cache_forces_new_instance(self):
        """clear_cache() should cause next create() to return a fresh instance."""
        runner1 = RunnerFactory.create("gemini")
        RunnerFactory.clear_cache()
        runner2 = RunnerFactory.create("gemini")

        assert runner1 is not runner2  # Different instances after cache clear
        assert isinstance(runner2, GeminiRunner)

    def test_create_caches_by_agent_name(self):
        """Cache key is the agent name; same name returns same object."""
        runner_a = RunnerFactory.create("gemini")
        runner_b = RunnerFactory.create("gemini")

        assert runner_a is runner_b
        assert isinstance(runner_a, GeminiRunner)

    def test_list_agents_returns_sorted_order(self):
        """list_agents() should return agents in sorted (alphabetical) order."""
        agents = RunnerFactory.list_agents()

        assert agents == sorted(agents)
