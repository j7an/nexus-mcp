# src/nexus_mcp/runners/gemini.py
"""Gemini CLI runner implementation.

Executes Google's Gemini CLI (https://github.com/google/generative-ai-python)
with JSON output format.

Command format:
    gemini -p "<prompt>" --output-format json [--model <model>] [--yolo]

Expected JSON response:
    {"response": "...", "stats": {...}}  # stats field is optional
"""

import asyncio
import contextlib
import json
from typing import Any, ClassVar

from nexus_mcp.config import get_agent_fallback_models
from nexus_mcp.exceptions import ParseError, RetryableError, SubprocessTimeoutError
from nexus_mcp.parser import extract_last_json_array, extract_last_json_object
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, ExecutionMode, LogEmitter, ProgressEmitter, PromptRequest


class GeminiRunner(AbstractRunner):
    """Runner for Google Gemini CLI.

    Builds commands in format:
        gemini -p <prompt> --output-format json [options]

    Parses JSON responses:
        {"response": "text", "stats": {...}}
    """

    AGENT_NAME = "gemini"
    _SUPPORTED_MODES: ClassVar[tuple[ExecutionMode, ...]] = ("default", "yolo")

    def build_command(self, request: PromptRequest) -> list[str]:
        """Build Gemini CLI command from request.

        Args:
            request: PromptRequest with prompt, model, execution_mode, file_refs.

        Returns:
            Command list: [cli_path, "-p", <prompt>, "--output-format", "json", ...]

        Command structure:
            1. Base: {cli_path} -p <prompt> (with file_refs appended if provided)
            2. Add --output-format json if capabilities.supports_json
            3. Add --model <model> (request.model > env default > CLI default)
            4. Add --yolo if execution_mode == "yolo"
        """
        command = [self.cli_path, "-p", self._build_prompt(request)]

        # Add --output-format json if supported by CLI version
        if self.capabilities.supports_json:
            command.extend(["--output-format", "json"])

        # Determine model: request.model > env default > CLI default
        model = request.model or self.default_model
        if model:
            command.extend(["--model", model])

        # Add execution mode flag
        match request.execution_mode:
            case "yolo":
                command.append("--yolo")
            case _:
                pass  # default: no auto-approve flags

        return command

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        """Parse Gemini CLI JSON output into AgentResponse.

        Args:
            stdout: JSON output from Gemini CLI (may contain Node.js warning prefix).
            stderr: Standard error output (not used for parsing).

        Returns:
            AgentResponse with:
                - agent: "gemini"
                - output: Stripped response text
                - raw_output: Original stdout
                - metadata: stats dict if present (flat, not nested)

        Raises:
            ParseError: If JSON is invalid, missing 'response' field,
                        or 'response' is not a string.

        Expected JSON format:
            {"response": "answer text", "stats": {...}}
            # stats field is optional

        Noisy stdout handling:
            Gemini CLI v0.29.0+ (Node.js-based) may prepend deprecation warnings and
            log lines before the JSON response. If json.loads() fails, falls back to
            _extract_last_json_object() which uses brace-depth matching to locate the
            JSON block within the mixed-content stdout.
        """
        # Parse JSON — fast path for clean stdout
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Fallback: extract JSON from noisy output (Node.js warnings, log lines)
            data = extract_last_json_object(stdout)
            if data is None:
                raise ParseError(
                    "Invalid JSON from Gemini CLI (stdout may contain non-JSON prefix)",
                    raw_output=stdout,
                ) from None

        # Validate 'response' field exists (guard against scalar JSON like null or 42)
        if not isinstance(data, dict) or "response" not in data:
            raise ParseError(
                "Missing 'response' field in Gemini CLI output",
                raw_output=stdout,
            )

        # Validate 'response' is a string
        response_text = data["response"]
        if not isinstance(response_text, str):
            raise ParseError(
                f"'response' field must be a string, got {type(response_text).__name__}",
                raw_output=stdout,
            )

        # Extract optional stats (flat metadata - stats dict IS the metadata)
        # Use 'or {}' rather than .get("stats", {}) so that explicit null ("stats": null)
        # is treated identically to an absent key — both yield an empty metadata dict.
        metadata = data.get("stats") or {}

        return AgentResponse(
            cli=self.AGENT_NAME,
            output=response_text.strip(),
            raw_output=stdout,
            metadata=metadata,
        )

    def _try_extract_error(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        command: list[str] | None = None,
    ) -> None:
        """Extract and raise a structured API error from stdout or stderr JSON.

        Looks for Gemini API error format: {"error": {"code": N, "message": "...", "status": "..."}}
        Tries stdout first (richer API error format); falls back to stderr JSON extraction.

        Args:
            stdout: Subprocess stdout to inspect first.
            stderr: Subprocess stderr used as fallback JSON source and passed through to error.
            returncode: Exit code (passed through to SubprocessError).
            command: The CLI command that was run (passed through to SubprocessError).

        Raises:
            SubprocessError: If stdout or stderr contains a Gemini API error object.
        """
        data = self._extract_error_payload(stdout, stderr)
        if data is None:
            return

        error = data["error"]  # _extract_error_payload guarantees this is a dict
        code = self._coerce_error_code(error.get("code", "unknown"))
        message = error.get("message", "unknown error")
        status = error.get("status", "")
        if code == 1 and message == "[object Object]":
            return
        error_msg = f"Gemini API error {code}: {message} ({status})"
        self._raise_structured_error(error_msg, code, stdout, stderr, returncode, command)

    # ------------------------------------------------------------------
    # Fallback-chain helpers
    # ------------------------------------------------------------------

    def _resolve_primary_model(self, request: PromptRequest) -> str | None:
        return request.model or self.default_model

    def _extract_error_payload(self, stdout: str, stderr: str) -> dict[str, Any] | None:
        """Locate a Gemini error JSON object in stdout (preferred) or stderr.

        Returns the parent dict whose ``error`` field is a dict, or None if no
        such payload exists.
        """
        data: dict[str, Any] | None = None
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                data = parsed

        if data is None:
            data = extract_last_json_array(stderr)
        if data is None:
            data = extract_last_json_object(stderr)

        if data is None:
            return None

        error = data.get("error")
        if not isinstance(error, dict):
            return None
        return data

    def _build_fallback_chain(self, primary_model: str) -> tuple[str, ...]:
        """Build the ordered fallback chain starting from the primary model.

        Deduplicates while preserving first-seen order so each model is tried
        at most once per runner-level attempt.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for model in (primary_model, *get_agent_fallback_models(self.AGENT_NAME)):
            if model not in seen:
                ordered.append(model)
                seen.add(model)
        return tuple(ordered)

    def _with_effective_model_metadata(
        self,
        response: AgentResponse,
        *,
        original_model: str,
        effective_model: str,
        fallback_attempt: int,
        fallback_reason: str | None,
    ) -> AgentResponse:
        """Stamp fallback bookkeeping onto a successful response."""
        metadata: dict[str, object] = {
            "effective_model": effective_model,
            "fallback_model_used": effective_model != original_model,
            "original_model": original_model,
        }
        if effective_model != original_model:
            metadata["fallback_attempt"] = fallback_attempt
            if fallback_reason:
                metadata["fallback_reason"] = fallback_reason
        return response.with_metadata(**metadata)

    def _retryable_reason(self, error: RetryableError | None) -> str | None:
        """Best-effort summary of why the previous model in the chain failed."""
        if error is None:
            return None
        data = self._extract_error_payload(error.stdout, error.stderr)
        if data is None:
            return str(error)
        status = data["error"].get("status")
        if isinstance(status, str) and status:
            return status
        return str(error)

    async def _execute(
        self, request: PromptRequest, emit: LogEmitter, progress: ProgressEmitter
    ) -> AgentResponse:
        """Walk the fallback model chain on retryable failures.

        For a single runner-level attempt, try each model in order. A
        ``RetryableError`` from one model triggers an immediate switch to the
        next; non-retryable errors propagate as before. The first success is
        returned with fallback metadata stamped on the response.
        """
        primary_model = self._resolve_primary_model(request)
        if primary_model is None:
            return await self._execute_single_attempt(request, emit, progress)

        chain = self._build_fallback_chain(primary_model)
        last_retryable: RetryableError | None = None

        for index, model in enumerate(chain, start=1):
            attempt_request = request.model_copy(update={"model": model})
            try:
                response = await self._execute_single_attempt(attempt_request, emit, progress)
                reason = self._retryable_reason(last_retryable) if last_retryable else None
                return self._with_effective_model_metadata(
                    response,
                    original_model=primary_model,
                    effective_model=model,
                    fallback_attempt=index,
                    fallback_reason=reason,
                )
            except RetryableError as error:
                last_retryable = error
                if index == len(chain):
                    raise
                await emit(
                    "warning",
                    f"Gemini retryable error on model {model}; "
                    f"switching to fallback model {chain[index]}",
                )
                continue
            except SubprocessTimeoutError:
                raise
            except asyncio.CancelledError:
                raise
        # Unreachable: loop always returns or raises — satisfies type checker
        raise AssertionError("unreachable: fallback loop exited without result or exception")
