# src/nexus_mcp/runners/base.py
"""Base classes for CLI agent runners.

Defines:
- CLIRunner: Protocol specifying the runner interface
- AbstractRunner: ABC implementing Template Method pattern

Template Method pattern in AbstractRunner._execute():
1. build_command(request) → list[str]
2. run_subprocess(command) → SubprocessResult
3. Check returncode != 0 → raise SubprocessError (fail fast)
4. parse_output(stdout, stderr) → AgentResponse
5. _apply_output_limit(response) → AgentResponse (truncate if needed)

AbstractRunner.run() wraps _execute() in a retry loop:
- Retries on RetryableError with exponential backoff + full jitter
- Non-retryable errors (SubprocessError, ParseError) propagate immediately
- max_attempts from request.max_retries or NEXUS_RETRY_MAX_ATTEMPTS env var
"""

import asyncio
import contextlib
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar, NoReturn, Protocol

from nexus_mcp.cli_detector import (
    CLICapabilities,
    detect_cli,
    get_cli_capabilities,
    get_cli_version,
)
from nexus_mcp.config import (
    get_agent_env,
    get_global_output_limit,
    get_global_timeout,
    get_retry_base_delay,
    get_retry_max_attempts,
    get_retry_max_delay,
)
from nexus_mcp.exceptions import CLINotFoundError, ParseError, RetryableError, SubprocessError
from nexus_mcp.process import run_subprocess
from nexus_mcp.types import AgentResponse, ExecutionMode, PromptRequest

logger = logging.getLogger(__name__)


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

    Subclasses must define:
    - AGENT_NAME: ClassVar[str]  (e.g., "gemini", "codex")

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

    AGENT_NAME: ClassVar[str]
    _SUPPORTED_MODES: ClassVar[tuple[ExecutionMode, ...]]
    _RETRYABLE_CODES: ClassVar[frozenset[int]] = frozenset({429, 503})

    def __init__(self) -> None:
        """Initialize runner with CLI detection and global configuration.

        Raises:
            CLINotFoundError: If the CLI binary is not found in PATH.
        """
        info = detect_cli(self.AGENT_NAME)
        if not info.found:
            raise CLINotFoundError(self.AGENT_NAME)
        version = get_cli_version(self.AGENT_NAME)
        self.capabilities: CLICapabilities = get_cli_capabilities(self.AGENT_NAME, version)
        self.timeout = get_global_timeout()
        self.base_delay = get_retry_base_delay()
        self.max_delay = get_retry_max_delay()
        self.default_max_attempts = get_retry_max_attempts()
        self.cli_path: str = (
            get_agent_env(self.AGENT_NAME, "PATH", default=self.AGENT_NAME) or self.AGENT_NAME
        )
        self.default_model: str | None = get_agent_env(self.AGENT_NAME, "MODEL")

    async def run(self, request: PromptRequest) -> AgentResponse:
        """Execute CLI agent with retry on transient errors.

        Wraps _execute() in a retry loop. RetryableError triggers exponential
        backoff with full jitter. All other exceptions propagate immediately.

        Args:
            request: Prompt request with agent, prompt, execution mode, etc.
                     request.max_retries overrides the env-var default when set.

        Returns:
            AgentResponse with parsed output and metadata.

        Raises:
            RetryableError: If all retry attempts are exhausted.
            SubprocessError: If CLI fails with a non-retryable error code.
            ParseError: If output parsing fails.
        """
        max_attempts = (
            request.max_retries if request.max_retries is not None else self.default_max_attempts
        )
        for attempt in range(max_attempts):
            try:
                return await self._execute(request)
            except RetryableError as e:
                if attempt == max_attempts - 1:
                    raise
                delay = self._compute_backoff(attempt, e.retry_after)
                logger.warning(
                    "Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_attempts,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
        # Unreachable: loop always returns or raises — satisfies type checker
        raise AssertionError("unreachable: retry loop exited without result or exception")

    async def _execute(self, request: PromptRequest) -> AgentResponse:
        """Execute CLI agent once using Template Method pattern.

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
            RetryableError: If _recover_from_error raises for a retryable error code.
            SubprocessError: If CLI fails (non-zero exit) or subprocess execution errors.
            ParseError: If output parsing fails (from parse_output()).
        """
        # Step 1: Build command
        command = self.build_command(request)

        # Step 2: Execute subprocess
        effective_timeout = request.timeout if request.timeout is not None else self.timeout
        result = await run_subprocess(command, timeout=effective_timeout)

        # Step 3: Error check with recovery attempt
        if result.returncode != 0:
            recovered = self._recover_from_error(
                result.stdout, result.stderr, result.returncode, command
            )
            if recovered:
                return self._apply_output_limit(recovered, request)
            raise SubprocessError(
                f"CLI command failed with exit code {result.returncode}",
                stderr=result.stderr,
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
            )

        # Step 4: Parse output
        response = self.parse_output(result.stdout, result.stderr)

        # Step 5: Apply output limiting
        return self._apply_output_limit(response, request)

    def _compute_backoff(self, attempt: int, retry_after: float | None) -> float:
        """Compute exponential backoff delay with full jitter.

        Uses AWS-recommended full jitter formula:
            delay = random.uniform(0, min(max_delay, base_delay * 2^attempt))

        If retry_after hint is provided, uses max(computed, retry_after) to
        respect the server's suggested wait time.

        Args:
            attempt: Zero-based attempt index (0 = first retry after first failure).
            retry_after: Optional server-suggested wait time in seconds.

        Returns:
            Delay in seconds to wait before the next attempt.
        """
        cap = min(self.max_delay, self.base_delay * (2**attempt))
        computed = random.uniform(0, cap)
        if retry_after is not None:
            return max(computed, retry_after)
        return computed

    def _apply_output_limit(self, response: AgentResponse, request: PromptRequest) -> AgentResponse:
        """Truncate output if it exceeds the configured byte limit.

        Args:
            response: Original response from parse_output()
            request: Prompt request, checked for per-request output_limit override.

        Returns:
            Response with truncated output if needed; metadata includes original/truncated sizes.
        """
        limit = (
            request.output_limit if request.output_limit is not None else get_global_output_limit()
        )
        output_bytes = response.output.encode("utf-8")
        output_size = len(output_bytes)

        if output_size <= limit:
            return response  # No truncation needed

        # Reserve space for truncation message (computed with actual sizes)
        suffix = f"\n\n[Output truncated: {output_size} bytes exceeds {limit} byte limit]"
        suffix_bytes = len(suffix.encode("utf-8"))
        content_limit = max(0, limit - suffix_bytes)

        # Truncate by bytes, then decode (handles multi-byte characters correctly)
        truncated_bytes = output_bytes[:content_limit]
        truncated_output = truncated_bytes.decode("utf-8", errors="ignore")
        truncated_output += suffix

        return response.model_copy(update={"output": truncated_output}).with_metadata(
            truncated=True,
            original_size_bytes=output_size,
            truncated_size_bytes=len(truncated_output.encode("utf-8")),
        )

    def _build_prompt(self, request: PromptRequest) -> str:
        """Build prompt string with optional file references appended.

        Args:
            request: Prompt request potentially containing file_refs.

        Returns:
            Prompt string with file references appended if present.
        """
        if not request.file_refs:
            return request.prompt
        file_list = "\n".join(f"- {path}" for path in request.file_refs)
        return f"{request.prompt}\n\nFile references:\n{file_list}"

    def _make_recovered_response(
        self, response: AgentResponse, returncode: int, stderr: str
    ) -> AgentResponse:
        """Stamp recovery metadata onto a response.

        Args:
            response: The successfully parsed AgentResponse from a failed subprocess call.
            returncode: The non-zero exit code of the subprocess.
            stderr: The subprocess stderr (preserved for caller inspection).

        Returns:
            A new AgentResponse with recovery metadata added.
        """
        return response.with_metadata(
            recovered_from_error=True,
            original_exit_code=returncode,
            stderr=stderr,
        )

    @staticmethod
    def _coerce_error_code(code: str | int | Any) -> str | int:
        """Coerce a string error code to int, falling back to original value.

        Args:
            code: Error code from CLI output, may be int or string representation of int.

        Returns:
            Integer if code is a string that parses as int, otherwise the original value.
        """
        if isinstance(code, str):
            with contextlib.suppress(ValueError):
                return int(code)
        return code

    def _raise_structured_error(
        self,
        error_msg: str,
        code: str | int,
        stdout: str,
        stderr: str,
        returncode: int,
        command: list[str] | None,
    ) -> NoReturn:
        """Raise RetryableError or SubprocessError based on error code.

        Args:
            error_msg: Human-readable error message.
            code: Error code (int checked against _RETRYABLE_CODES, str always non-retryable).
            stdout: Subprocess stdout (forwarded to exception).
            stderr: Subprocess stderr (forwarded to exception).
            returncode: Exit code (forwarded to exception).
            command: The CLI command that was run (forwarded to exception).

        Raises:
            RetryableError: If code is an int in _RETRYABLE_CODES.
            SubprocessError: Otherwise.
        """
        if isinstance(code, int) and code in self._RETRYABLE_CODES:
            raise RetryableError(
                error_msg, stderr=stderr, stdout=stdout, returncode=returncode, command=command
            )
        raise SubprocessError(
            error_msg, stderr=stderr, stdout=stdout, returncode=returncode, command=command
        )

    def _try_extract_error(
        self, stdout: str, stderr: str, returncode: int, command: list[str] | None = None
    ) -> None:
        """Hook for subclasses to raise structured errors from CLI output.

        Called by _recover_from_error when parse_output fails.
        Default: no-op. Override to inspect stdout/stderr for agent-specific
        error formats and raise SubprocessError with structured details.

        Args:
            stdout: Subprocess stdout to inspect.
            stderr: Subprocess stderr to inspect.
            returncode: Exit code (for forwarding to SubprocessError if raised).
            command: The CLI command that was run (for forwarding to SubprocessError).
        """
        return

    def _recover_from_error(
        self, stdout: str, stderr: str, returncode: int, command: list[str] | None = None
    ) -> AgentResponse | None:
        """Attempt to recover from subprocess error.

        Tries to parse stdout as valid output even on non-zero exit code.
        If parsing fails, calls _try_extract_error() to surface structured
        errors from stdout/stderr before falling through to the generic error.

        Args:
            stdout: Subprocess stdout
            stderr: Subprocess stderr
            returncode: Exit code
            command: The CLI command that was run (may be forwarded to SubprocessError).

        Returns:
            Recovered AgentResponse or None if recovery not possible.

        Raises:
            SubprocessError: If _try_extract_error detects a structured error in
                stdout/stderr (e.g., API error JSON with code and message).
        """
        try:
            response = self.parse_output(stdout, stderr)
            return self._make_recovered_response(response, returncode, stderr)
        except ParseError:
            # Try to extract structured API error; raises SubprocessError if found
            self._try_extract_error(stdout, stderr, returncode, command)
            return None  # Falls through to base.py's generic SubprocessError

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
