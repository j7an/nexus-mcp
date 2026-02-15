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
5. _apply_output_limit(response) → AgentResponse (truncate if needed)
"""

import tempfile
from abc import ABC, abstractmethod
from typing import Protocol

from nexus_mcp.config import get_global_output_limit, get_global_timeout
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
    3. Error check with recovery attempt
    4. Parse output
    5. Apply output limiting
    """

    def __init__(self) -> None:
        """Initialize runner with global configuration."""
        self.timeout = get_global_timeout()

    async def run(self, request: PromptRequest) -> AgentResponse:
        """Execute CLI agent using Template Method pattern.

        Template steps:
        1. build_command(request) → command list
        2. run_subprocess(command) → SubprocessResult
        3. Check returncode != 0 → try _recover_from_error() or raise SubprocessError
        4. parse_output(stdout, stderr) → AgentResponse
        5. _apply_output_limit(response) → AgentResponse (truncate if needed)

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
        result = await run_subprocess(command, timeout=self.timeout)

        # Step 3: Error check with recovery attempt
        if result.returncode != 0:
            recovered = self._recover_from_error(result.stdout, result.stderr, result.returncode)
            if recovered:
                return self._apply_output_limit(recovered)
            raise SubprocessError(
                f"CLI command failed with exit code {result.returncode}",
                stderr=result.stderr,
                command=command,
                returncode=result.returncode,
            )

        # Step 4: Parse output
        response = self.parse_output(result.stdout, result.stderr)

        # Step 5: Apply output limiting
        return self._apply_output_limit(response)

    def _apply_output_limit(self, response: AgentResponse) -> AgentResponse:
        """Truncate output if exceeds limit, save full output to temp file.

        Args:
            response: Original response from parse_output()

        Returns:
            Response with truncated output if needed, metadata includes temp file path
        """
        limit = get_global_output_limit()
        output_bytes = response.output.encode("utf-8")
        output_size = len(output_bytes)

        if output_size <= limit:
            return response  # No truncation needed

        # Save full output to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            delete=False,
            prefix="nexus_mcp_output_",
            suffix=".txt",
        ) as temp_file:
            temp_file.write(response.output)
            temp_file_name = temp_file.name

        # Reserve space for truncation message
        suffix = "\n\n[Output truncated - see full output at temp file]"
        suffix_bytes = len(suffix.encode("utf-8"))
        content_limit = max(0, limit - suffix_bytes)

        # Truncate by bytes, then decode (handles multi-byte characters correctly)
        truncated_bytes = output_bytes[:content_limit]
        truncated_output = truncated_bytes.decode("utf-8", errors="ignore")
        truncated_output += suffix

        # Update metadata
        metadata = response.metadata.copy()
        metadata["truncated"] = True
        metadata["full_output_path"] = temp_file_name
        metadata["original_size_bytes"] = output_size
        metadata["truncated_size_bytes"] = len(truncated_output.encode("utf-8"))

        return AgentResponse(
            agent=response.agent,
            output=truncated_output,
            raw_output=response.raw_output,
            metadata=metadata,
        )

    def _recover_from_error(
        self, stdout: str, stderr: str, returncode: int
    ) -> AgentResponse | None:
        """Attempt to recover from subprocess error.

        Default implementation returns None (no recovery). Subclasses can override
        to implement CLI-specific recovery strategies.

        Args:
            stdout: Subprocess stdout
            stderr: Subprocess stderr
            returncode: Exit code

        Returns:
            Recovered AgentResponse or None if recovery not possible
        """
        return None

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
