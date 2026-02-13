# src/nexus_mcp/runners/base.py
"""Base classes for CLI agent runners.

Defines:
- CLIRunner: Protocol specifying the runner interface
- AbstractRunner: ABC implementing Template Method pattern

Template Method pattern in AbstractRunner.run():
1. build_command(request) → list[str]
2. run_subprocess(command) → SubprocessResult
3. Check returncode != 0 → raise SubprocessError (fail fast)
4. parse_output(stdout, stderr) → AgentResponse
"""

from abc import ABC, abstractmethod
from typing import Protocol

from nexus_mcp.exceptions import SubprocessError
from nexus_mcp.process import run_subprocess
from nexus_mcp.types import AgentResponse, PromptRequest


class CLIRunner(Protocol):
    """Protocol defining the interface for CLI agent runners.

    All runners must implement async run() method that takes a PromptRequest
    and returns an AgentResponse.
    """

    async def run(self, request: PromptRequest) -> AgentResponse:
        """Execute CLI agent with given request.

        Args:
            request: Prompt request containing agent, prompt, and execution settings.

        Returns:
            AgentResponse containing agent output and metadata.

        Raises:
            SubprocessError: If CLI execution fails.
            ParseError: If CLI output cannot be parsed.
        """
        ...


class AbstractRunner(ABC):
    """Abstract base class implementing Template Method pattern for CLI runners.

    Subclasses must implement:
    - build_command(): Construct CLI command from PromptRequest
    - parse_output(): Parse CLI stdout/stderr into AgentResponse

    The run() method orchestrates the execution flow (Template Method):
    1. Build command
    2. Execute subprocess
    3. Fail fast on non-zero return code
    4. Parse output
    """

    async def run(self, request: PromptRequest) -> AgentResponse:
        """Execute CLI agent using Template Method pattern.

        Template steps:
        1. build_command(request) → command list
        2. run_subprocess(command) → SubprocessResult
        3. Check returncode != 0 → raise SubprocessError (fail fast before parsing)
        4. parse_output(stdout, stderr) → AgentResponse

        Args:
            request: Prompt request with agent, prompt, execution mode, etc.

        Returns:
            AgentResponse with parsed output and metadata.

        Raises:
            SubprocessError: If CLI fails (non-zero exit) or subprocess execution errors.
            ParseError: If output parsing fails (from parse_output()).
        """
        # Step 1: Build command
        command = self.build_command(request)

        # Step 2: Execute subprocess
        result = await run_subprocess(command)

        # Step 3: Fail fast on CLI errors (before parsing)
        if result.returncode != 0:
            raise SubprocessError(
                f"CLI command failed with exit code {result.returncode}",
                stderr=result.stderr,
                command=command,
                returncode=result.returncode,
            )

        # Step 4: Parse output
        return self.parse_output(result.stdout, result.stderr)

    @abstractmethod
    def build_command(self, request: PromptRequest) -> list[str]:
        """Build CLI command from request.

        Args:
            request: Prompt request containing prompt, model, execution_mode, etc.

        Returns:
            Command as list of arguments (e.g., ["gemini", "-p", "prompt"]).
        """
        ...

    @abstractmethod
    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        """Parse CLI output into AgentResponse.

        Args:
            stdout: Standard output from CLI.
            stderr: Standard error from CLI.

        Returns:
            AgentResponse with parsed output.

        Raises:
            ParseError: If output cannot be parsed.
        """
        ...
