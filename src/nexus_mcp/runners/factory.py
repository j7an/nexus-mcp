# src/nexus_mcp/runners/factory.py
"""Factory for creating CLI agent runners.

Uses match/case pattern matching (Python 3.12+) to dispatch agent names
to appropriate runner classes.

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
    """

    _REGISTRY: dict[str, type[AbstractRunner]] = {
        "gemini": GeminiRunner,
    }

    @staticmethod
    def create(agent: str) -> AbstractRunner:
        """Create a runner instance for the specified agent.

        Args:
            agent: Agent name (case-sensitive: "gemini", "codex", "claude").

        Returns:
            AbstractRunner instance for the agent.

        Raises:
            UnsupportedAgentError: If agent name is not recognized.

        Example:
            runner = RunnerFactory.create("gemini")  # → GeminiRunner instance
        """
        runner_class = RunnerFactory._REGISTRY.get(agent)
        if runner_class is None:
            raise UnsupportedAgentError(agent)
        return runner_class()

    @staticmethod
    def list_agents() -> list[str]:
        """List all supported agent names in sorted order.

        Returns:
            Sorted list of agent names that can be passed to create().

        Example:
            RunnerFactory.list_agents()  # → ["gemini"]
        """
        return sorted(RunnerFactory._REGISTRY)
