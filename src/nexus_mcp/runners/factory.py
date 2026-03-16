# src/nexus_mcp/runners/factory.py
"""Factory for creating and caching CLI agent runners.

Uses a registry dict for O(1) agent lookup and caches runner instances so
that blocking CLI detection (subprocess.run in __init__) only runs once per
agent per process lifetime.

Usage:
    runner = RunnerFactory.create("gemini")
    response = await runner.run(request)
"""

from typing import ClassVar

from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.runners.claude import ClaudeRunner
from nexus_mcp.runners.codex import CodexRunner
from nexus_mcp.runners.gemini import GeminiRunner
from nexus_mcp.runners.opencode import OpenCodeRunner


class RunnerFactory:
    """Factory for creating CLI agent runners.

    Creates appropriate runner instances based on agent name.
    Uses a registry dict for O(1) lookup and a single source of truth for
    supported agents (add to _REGISTRY to support new CLIs).

    Runner instances are cached after first construction so that repeated
    create() calls do not re-invoke blocking CLI detection in __init__.
    Call clear_cache() in tests to prevent instance leakage between cases.
    """

    _REGISTRY: ClassVar[dict[str, type[AbstractRunner]]] = {
        ClaudeRunner.AGENT_NAME: ClaudeRunner,
        CodexRunner.AGENT_NAME: CodexRunner,
        GeminiRunner.AGENT_NAME: GeminiRunner,
        OpenCodeRunner.AGENT_NAME: OpenCodeRunner,
    }

    _instances: ClassVar[dict[str, AbstractRunner]] = {}

    @classmethod
    def create(cls, agent: str) -> AbstractRunner:
        """Return cached runner instance for the specified agent.

        Constructs and caches the runner on first call; subsequent calls with
        the same agent name return the same object. This avoids repeated
        blocking subprocess calls in runner __init__ methods.

        Args:
            agent: CLI runner name (case-sensitive: "claude", "codex", "gemini", "opencode").

        Returns:
            AbstractRunner instance for the agent.

        Raises:
            UnsupportedAgentError: If agent name is not recognized.

        Example:
            runner = RunnerFactory.create("gemini")  # → GeminiRunner instance
        """
        cached = cls._instances.get(agent)
        if cached is not None:
            return cached
        runner_class = cls._REGISTRY.get(agent)
        if runner_class is None:
            raise UnsupportedAgentError(agent)
        instance = runner_class()
        cls._instances[agent] = instance
        return instance

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the runner instance cache.

        Forces the next create() call to construct a fresh runner instance.
        Use in test teardown to prevent cached runners (built with mocked CLI
        detection) from leaking into subsequent tests.

        Example:
            RunnerFactory.clear_cache()  # reset between tests
        """
        cls._instances.clear()

    @classmethod
    def list_clis(cls) -> list[str]:
        """Return sorted list of registered runner names.

        Returns:
            Sorted list of CLI names that can be passed to create() or get_runner_class().

        Example:
            RunnerFactory.list_clis()  # → ["claude", "codex", "gemini", "opencode"]
        """
        return sorted(cls._REGISTRY)

    @classmethod
    def get_runner_class(cls, name: str) -> type[AbstractRunner]:
        """Return the runner class for the given name without instantiation.

        Args:
            name: Agent name (case-sensitive).

        Returns:
            The AbstractRunner subclass registered under that name.

        Raises:
            UnsupportedAgentError: If name is not in the registry.

        Example:
            cls = RunnerFactory.get_runner_class("gemini")  # → GeminiRunner
        """
        try:
            return cls._REGISTRY[name]
        except KeyError:
            raise UnsupportedAgentError(name) from None
