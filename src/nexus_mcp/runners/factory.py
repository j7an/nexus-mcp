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
from nexus_mcp.runners.opencode_server import OpenCodeServerRunner


class RunnerFactory:
    """Factory for creating CLI agent runners.

    Creates appropriate runner instances based on CLI name.
    Uses a registry dict for O(1) lookup and a single source of truth for
    supported CLIs (add to _REGISTRY to support new CLIs).

    Runner instances are cached after first construction so that repeated
    create() calls do not re-invoke blocking CLI detection in __init__.
    Call clear_cache() in tests to prevent instance leakage between cases.
    """

    _REGISTRY: ClassVar[dict[str, type[AbstractRunner]]] = {
        ClaudeRunner.AGENT_NAME: ClaudeRunner,
        CodexRunner.AGENT_NAME: CodexRunner,
        GeminiRunner.AGENT_NAME: GeminiRunner,
        OpenCodeRunner.AGENT_NAME: OpenCodeRunner,
        OpenCodeServerRunner.AGENT_NAME: OpenCodeServerRunner,
    }

    _instances: ClassVar[dict[str, AbstractRunner]] = {}

    @classmethod
    def create(cls, name: str) -> AbstractRunner:
        """Return cached runner instance for the specified CLI.

        Constructs and caches the runner on first call; subsequent calls with
        the same CLI name return the same object. This avoids repeated
        blocking subprocess calls in runner __init__ methods.

        Args:
            name: CLI runner name (case-sensitive: "claude", "codex", "gemini",
                "opencode", "opencode_server").

        Returns:
            AbstractRunner instance for the name.

        Raises:
            UnsupportedAgentError: If name is not recognized.

        Example:
            runner = RunnerFactory.create("gemini")  # → GeminiRunner instance
        """
        cached = cls._instances.get(name)
        if cached is not None:
            return cached
        runner_class = cls._REGISTRY.get(name)
        if runner_class is None:
            raise UnsupportedAgentError(name)
        instance = runner_class()
        cls._instances[name] = instance
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
        """Return sorted list of registered CLI runner names.

        Returns:
            Sorted list of CLI names that can be passed to create() or get_runner_class().

        Example:
            RunnerFactory.list_clis()  # → ["claude", "codex", ..., "opencode_server"]
        """
        return sorted(cls._REGISTRY)

    @classmethod
    def get_runner_class(cls, name: str) -> type[AbstractRunner]:
        """Return the runner class for the given name without instantiation.

        Args:
            name: CLI name (case-sensitive).

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
