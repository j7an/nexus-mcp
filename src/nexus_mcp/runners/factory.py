# src/nexus_mcp/runners/factory.py
"""Factory for creating CLI agent runners.

Uses match/case pattern matching (Python 3.12+) to dispatch agent names
to appropriate runner classes.

Usage:
    factory = RunnerFactory()
    runner = factory.create("gemini")
    response = await runner.run(request)
"""

from nexus_mcp.exceptions import UnsupportedAgentError
from nexus_mcp.runners.base import CLIRunner
from nexus_mcp.runners.gemini import GeminiRunner


class RunnerFactory:
    """Factory for creating CLI agent runners.

    Creates appropriate runner instances based on agent name.
    Uses match/case for clean agent dispatch.
    """

    def create(self, agent: str) -> CLIRunner:
        """Create a runner instance for the specified agent.

        Args:
            agent: Agent name (case-sensitive: "gemini", "codex", "claude").

        Returns:
            CLIRunner instance for the agent.

        Raises:
            UnsupportedAgentError: If agent name is not recognized.

        Example:
            factory = RunnerFactory()
            runner = factory.create("gemini")  # → GeminiRunner instance
        """
        match agent:
            case "gemini":
                return GeminiRunner()
            case _:
                raise UnsupportedAgentError(agent)

    def list_agents(self) -> list[str]:
        """List all supported agent names.

        Returns:
            List of agent names that can be passed to create().

        Example:
            factory.list_agents()  # → ["gemini"]
        """
        return ["gemini"]
