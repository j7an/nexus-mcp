"""Custom exception hierarchy for nexus-mcp.

All exceptions inherit from NexusMCPError, which inherits from Exception.
Each concrete exception stores context attributes and produces a human-readable message.
"""


class NexusMCPError(Exception):
    """Base exception for all nexus-mcp errors."""


class CLIProcessError(NexusMCPError):
    """Raised when a CLI subprocess exits with a non-zero return code."""

    def __init__(self, returncode: int, stderr: str, command: str) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.command = command
        super().__init__(
            f"CLI process failed: command={command!r}, returncode={returncode}, stderr={stderr!r}"
        )


class CLITimeoutError(NexusMCPError):
    """Raised when a CLI subprocess exceeds its timeout."""

    def __init__(self, timeout: float, command: str) -> None:
        self.timeout = timeout
        self.command = command
        super().__init__(f"CLI process timed out after {timeout}s: command={command!r}")


class ParseError(NexusMCPError):
    """Raised when CLI output cannot be parsed into the expected format."""

    def __init__(self, raw_output: str, agent: str) -> None:
        self.raw_output = raw_output
        self.agent = agent
        super().__init__(f"Failed to parse {agent} output: {raw_output!r}")


class AgentNotFoundError(NexusMCPError):
    """Raised when a requested agent is not registered."""

    def __init__(self, agent: str, available: list[str]) -> None:
        self.agent = agent
        self.available = available
        available_str = ", ".join(available)
        super().__init__(f"Agent {agent!r} not found. Available agents: {available_str}")
