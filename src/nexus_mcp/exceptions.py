"""Custom exception hierarchy for nexus-mcp.

All exceptions inherit from NexusMCPError, which inherits from Exception.
Each concrete exception stores context attributes for debugging.
"""


class NexusMCPError(Exception):
    """Base exception for all nexus-mcp errors.

    All custom exceptions in this package inherit from this base class.
    """


class SubprocessError(NexusMCPError):
    """Subprocess execution failed.

    Raised when a CLI subprocess exits with an error or encounters execution issues.

    Attributes:
        stderr: Standard error output from the subprocess.
        command: The command that was executed (as a list of arguments).
        returncode: The exit code returned by the subprocess.
    """

    def __init__(
        self,
        message: str,
        stderr: str = "",
        command: list[str] | None = None,
        returncode: int | None = None,
    ):
        """Initialize SubprocessError.

        Args:
            message: Human-readable error description.
            stderr: Standard error output from the subprocess (default: "").
            command: The command that failed (default: None).
            returncode: The subprocess exit code (default: None).
        """
        super().__init__(message)
        self.stderr = stderr
        self.command = command
        self.returncode = returncode


class ParseError(NexusMCPError):
    """Failed to parse CLI output.

    Raised when CLI output cannot be parsed into the expected format (e.g., invalid JSON).

    Attributes:
        raw_output: The unparseable output received from the CLI.
    """

    def __init__(self, message: str, raw_output: str = ""):
        """Initialize ParseError.

        Args:
            message: Human-readable error description.
            raw_output: The raw output that could not be parsed (default: "").
        """
        super().__init__(message)
        self.raw_output = raw_output


class UnsupportedAgentError(NexusMCPError):
    """Agent not recognized by RunnerFactory.

    Raised when attempting to create a runner for an unknown agent.

    Attributes:
        agent: The name of the unsupported agent.
    """

    def __init__(self, agent: str):
        """Initialize UnsupportedAgentError.

        Args:
            agent: The name of the unsupported agent.
        """
        super().__init__(f"Unsupported agent: {agent}")
        self.agent = agent


class SubprocessTimeoutError(SubprocessError):
    """Subprocess exceeded timeout.

    Raised when a CLI subprocess runs longer than the allowed timeout duration.
    Inherits all attributes from SubprocessError and adds timeout information.

    Attributes:
        timeout: The timeout duration in seconds that was exceeded.
        stderr: Inherited from SubprocessError.
        command: Inherited from SubprocessError.
        returncode: Inherited from SubprocessError.
    """

    def __init__(
        self,
        message: str,
        *,
        timeout: float,
        stderr: str = "",
        command: list[str] | None = None,
        returncode: int | None = None,
    ):
        """Initialize SubprocessTimeoutError.

        Args:
            message: Human-readable error description.
            timeout: The timeout duration in seconds (keyword-only).
            stderr: Standard error output from the subprocess (default: "").
            command: The command that timed out (default: None).
            returncode: The subprocess exit code (default: None).
        """
        super().__init__(message, stderr=stderr, command=command, returncode=returncode)
        self.timeout = timeout


class CLINotFoundError(NexusMCPError):
    """CLI binary not found in PATH.

    Raised when a required CLI tool is not installed or not in PATH.

    Attributes:
        cli_name: The name of the CLI that was not found.
    """

    def __init__(self, cli_name: str):
        """Initialize CLINotFoundError.

        Args:
            cli_name: The name of the CLI binary that was not found.
        """
        super().__init__(f"CLI not found in PATH: {cli_name}")
        self.cli_name = cli_name


class ConfigurationError(NexusMCPError):
    """Configuration validation failed.

    Raised when environment variable or configuration values are invalid.

    Attributes:
        config_key: The configuration key that failed validation (optional).
    """

    def __init__(self, message: str, config_key: str | None = None):
        """Initialize ConfigurationError.

        Args:
            message: Human-readable error description.
            config_key: The configuration key that failed (optional).
        """
        super().__init__(message)
        self.config_key = config_key
