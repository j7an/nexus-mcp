# src/nexus_mcp/runners/factory.py
"""Factory for creating and caching CLI agent runners.

Uses a registry dict for O(1) agent lookup and caches runner instances so
that blocking CLI detection (subprocess.run in __init__) only runs once per
agent per process lifetime.

Usage:
    runner = RunnerFactory.create("gemini")
    response = await runner.run(request)
"""

from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.runners.gemini import GeminiRunner


class RunnerFactory:
    """Factory for creating CLI agent runners.

    Creates appropriate runner instances based on agent name.
    Uses a registry dict for O(1) lookup and a single source of truth for
    supported agents (add to _REGISTRY to support new CLIs).

    Runner instances are cached after first construction so that repeated
    create() calls do not re-invoke blocking CLI detection in __init__.
    Call clear_cache() in tests to prevent instance leakage between cases.
    """

    _REGISTRY: dict[str, type[AbstractRunner]] = {
        GeminiRunner.AGENT_NAME: GeminiRunner,
    }

    _instances: dict[str, AbstractRunner] = {}

    @staticmethod
    def create(agent: str) -> AbstractRunner:
        """Return cached runner instance for the specified agent.

        Constructs and caches the runner on first call; subsequent calls with
        the same agent name return the same object. This avoids repeated
        blocking subprocess calls in runner __init__ methods.

        Args:
            agent: Agent name (case-sensitive: "gemini", "codex", "claude").

        Returns:
            AbstractRunner instance for the agent.

        Raises:
            UnsupportedAgentError: If agent name is not recognized.

        Example:
            runner = RunnerFactory.create("gemini")  # → GeminiRunner instance
        """
        cached = RunnerFactory._instances.get(agent)
        if cached is not None:
            return cached
        runner_class = RunnerFactory._REGISTRY.get(agent)
        if runner_class is None:
            raise UnsupportedAgentError(agent)
        instance = runner_class()
        RunnerFactory._instances[agent] = instance
        return instance

    @staticmethod
    def clear_cache() -> None:
        """Clear the runner instance cache.

        Forces the next create() call to construct a fresh runner instance.
        Use in test teardown to prevent cached runners (built with mocked CLI
        detection) from leaking into subsequent tests.

        Example:
            RunnerFactory.clear_cache()  # reset between tests
        """
        RunnerFactory._instances.clear()

    @staticmethod
    def list_agents() -> list[str]:
        """List all supported agent names in sorted order.

        Returns:
            Sorted list of agent names that can be passed to create().

        Example:
            RunnerFactory.list_agents()  # → ["gemini"]
        """
        return sorted(RunnerFactory._REGISTRY)
