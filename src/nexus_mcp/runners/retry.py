"""Retry mixin for CLI agent runners.

Provides RetryMixin with exponential backoff + full jitter retry logic.
AbstractRunner inherits this mixin to wrap _execute() in a retry loop.

Backoff formula (AWS-recommended full jitter):
    delay = random.uniform(0, min(max_delay, base_delay * 2^attempt))
"""

__all__ = ["RetryMixin"]

import asyncio
import logging
import random
from typing import TYPE_CHECKING

from nexus_mcp.exceptions import RetryableError
from nexus_mcp.types import LogEmitter, LogLevel, ProgressEmitter, PromptRequest

if TYPE_CHECKING:
    from nexus_mcp.types import AgentResponse

logger = logging.getLogger(__name__)


async def _default_log_emitter(level: LogLevel, message: str) -> None:
    """Fallback emitter: delegates to Python's stdlib logger.

    Used when no MCP-aware emitter is provided (direct runner usage, tests).
    """
    getattr(logger, level)(message)


async def _noop_progress(progress: float, total: float, message: str) -> None:
    """No-op progress emitter for direct runner usage and tests.

    Used when no MCP-aware progress emitter is provided.
    """


class RetryMixin:
    """Mixin providing retry-with-backoff for runner execution.

    Expects the concrete class to define:
    - async _execute(request, emit, progress) -> AgentResponse
    - self.default_max_attempts: int
    - self.base_delay: float
    - self.max_delay: float
    """

    # Declared for mypy — concrete values set by AbstractRunner.__init__
    default_max_attempts: int
    base_delay: float
    max_delay: float

    async def _execute(
        self, request: PromptRequest, emit: LogEmitter, progress: ProgressEmitter
    ) -> "AgentResponse":
        """Execute CLI agent once (implemented by AbstractRunner)."""
        raise NotImplementedError  # pragma: no cover

    async def run(
        self,
        request: PromptRequest,
        emitter: LogEmitter | None = None,
        progress: ProgressEmitter | None = None,
    ) -> "AgentResponse":
        """Execute CLI agent with retry on transient errors.

        Wraps _execute() in a retry loop. RetryableError triggers exponential
        backoff with full jitter. All other exceptions propagate immediately.

        Args:
            request: Prompt request with agent, prompt, execution mode, etc.
                     request.max_retries overrides the env-var default when set.
            emitter: Optional log emitter. When None, uses _default_log_emitter.
            progress: Optional progress emitter for structured progress reporting.
                      When None, uses _noop_progress.

        Returns:
            AgentResponse with parsed output and metadata.

        Raises:
            RetryableError: If all retry attempts are exhausted.
            SubprocessError: If CLI fails with a non-retryable error code.
            ParseError: If output parsing fails.
        """
        emit = emitter or _default_log_emitter
        report = progress or _noop_progress
        max_attempts = (
            request.max_retries if request.max_retries is not None else self.default_max_attempts
        )
        # Resolve per-request delay overrides once before the loop (concurrency-safe: avoids
        # mutating self.base_delay/max_delay which are shared across concurrent requests).
        # IMPORTANT: use `is not None`, NOT `or` — 0.0 is a valid value (instant backoff).
        effective_base_delay = (
            request.retry_base_delay if request.retry_base_delay is not None else self.base_delay
        )
        effective_max_delay = (
            request.retry_max_delay if request.retry_max_delay is not None else self.max_delay
        )
        for attempt in range(max_attempts):
            await report(attempt + 1, max_attempts, f"Attempt {attempt + 1}/{max_attempts}")
            try:
                return await self._execute(request, emit, report)
            except RetryableError as e:
                if attempt == max_attempts - 1:
                    raise
                delay = self._compute_backoff(
                    attempt, e.retry_after, effective_base_delay, effective_max_delay
                )
                await emit(
                    "warning",
                    f"Retryable error (attempt {attempt + 1}/{max_attempts}),"
                    f" retrying in {delay:.1f}s: {e}",
                )
                await report(
                    attempt + 1,
                    max_attempts,
                    f"Waiting {delay:.1f}s before retry {attempt + 2}/{max_attempts}",
                )
                await asyncio.sleep(delay)
        # Unreachable: loop always returns or raises — satisfies type checker
        raise AssertionError("unreachable: retry loop exited without result or exception")

    def _compute_backoff(
        self,
        attempt: int,
        retry_after: float | None,
        base_delay: float | None = None,
        max_delay: float | None = None,
    ) -> float:
        """Compute exponential backoff delay with full jitter.

        Uses AWS-recommended full jitter formula:
            delay = random.uniform(0, min(max_delay, base_delay * 2^attempt))

        If retry_after hint is provided, uses max(computed, retry_after) to
        respect the server's suggested wait time.

        Args:
            attempt: Zero-based attempt index (0 = first retry after first failure).
            retry_after: Optional server-suggested wait time in seconds.
            base_delay: Override base delay (None falls back to self.base_delay).
            max_delay: Override max delay cap (None falls back to self.max_delay).

        Returns:
            Delay in seconds to wait before the next attempt.
        """
        bd = base_delay if base_delay is not None else self.base_delay
        md = max_delay if max_delay is not None else self.max_delay
        cap = min(md, bd * (2**attempt))
        computed = random.uniform(0, cap)
        if retry_after is not None:
            return max(computed, retry_after)
        return computed
