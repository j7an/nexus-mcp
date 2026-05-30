# tests/unit/runners/test_factory.py
"""Tests for RunnerFactory.

Tests verify:
- create("codex") → CodexRunner instance
- create("claude") → ClaudeRunner instance
- create("opencode") → OpenCodeRunner instance
- create("gemini") → UnsupportedAgentError
- create("unknown") → UnsupportedAgentError
- list_clis() → ["claude", "codex", "opencode", "opencode_server"]
- create() returns cached instance for same agent
- clear_cache() forces fresh instance on next create()
"""

import pytest

from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.runners.claude import ClaudeRunner
from nexus_mcp.runners.codex import CodexRunner
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.runners.opencode import OpenCodeRunner
from nexus_mcp.runners.opencode_server import OpenCodeServerRunner


class TestRunnerFactory:
    """Test RunnerFactory agent creation and listing."""

    def test_create_codex_runner(self):
        """create("codex") should return CodexRunner instance."""
        runner = RunnerFactory.create("codex")

        assert isinstance(runner, CodexRunner)

    def test_create_unknown_agent_raises_error(self):
        """create("unknown") should raise UnsupportedAgentError."""
        with pytest.raises(UnsupportedAgentError) as exc_info:
            RunnerFactory.create("unknown-agent")

        assert exc_info.value.agent == "unknown-agent"
        assert "Unsupported agent: unknown-agent" in str(exc_info.value)

    def test_create_gemini_is_unsupported(self):
        """create("gemini") should raise UnsupportedAgentError."""
        with pytest.raises(UnsupportedAgentError) as exc_info:
            RunnerFactory.create("gemini")

        assert exc_info.value.agent == "gemini"
        assert "Unsupported agent: gemini" in str(exc_info.value)

    def test_create_case_sensitive(self):
        """Agent names are case-sensitive."""
        with pytest.raises(UnsupportedAgentError):
            RunnerFactory.create("Codex")

    def test_create_claude_runner(self):
        """create("claude") should return ClaudeRunner instance."""
        runner = RunnerFactory.create("claude")

        assert isinstance(runner, ClaudeRunner)

    def test_create_opencode_runner(self):
        """create("opencode") should return OpenCodeRunner instance."""
        runner = RunnerFactory.create("opencode")

        assert isinstance(runner, OpenCodeRunner)

    def test_list_clis_returns_supported_agents(self):
        """list_clis() should return list of supported agent names."""
        agents = RunnerFactory.list_clis()

        assert agents == ["claude", "codex", "opencode", "opencode_server"]

    def test_create_returns_cached_instance(self):
        """create() should return the same cached instance for the same agent."""
        runner1 = RunnerFactory.create("codex")
        runner2 = RunnerFactory.create("codex")

        assert runner1 is runner2  # Same cached instance

    def test_clear_cache_forces_new_instance(self):
        """clear_cache() should cause next create() to return a fresh instance."""
        runner1 = RunnerFactory.create("codex")
        RunnerFactory.clear_cache()
        runner2 = RunnerFactory.create("codex")

        assert runner1 is not runner2  # Different instances after cache clear
        assert isinstance(runner2, CodexRunner)

    def test_list_clis_returns_sorted_order(self):
        """list_clis() should return agents in sorted (alphabetical) order."""
        agents = RunnerFactory.list_clis()

        assert agents == sorted(agents)


class TestRunnerFactoryNewMethods:
    def test_list_clis_returns_sorted_names(self):
        assert RunnerFactory.list_clis() == [
            "claude",
            "codex",
            "opencode",
            "opencode_server",
        ]

    def test_get_runner_class_returns_class(self):
        cls = RunnerFactory.get_runner_class("codex")
        assert cls is CodexRunner

    def test_get_runner_class_unknown_raises(self):
        with pytest.raises(UnsupportedAgentError):
            RunnerFactory.get_runner_class("unknown")

    def test_opencode_server_is_registered(self):
        assert "opencode_server" in RunnerFactory.list_clis()

    def test_opencode_server_class(self):
        cls = RunnerFactory.get_runner_class("opencode_server")
        assert cls is OpenCodeServerRunner
