# src/nexus_mcp/runners/gemini.py
"""Gemini CLI runner implementation.

Executes Google's Gemini CLI (https://github.com/google/generative-ai-python)
with JSON output format.

Command format:
    gemini -p "<prompt>" --output-format json [--model <model>] [--yolo]

Expected JSON response:
    {"response": "...", "stats": {...}}  # stats field is optional
"""

import contextlib
import json
from typing import Any, ClassVar

from nexus_mcp.exceptions import ParseError
from nexus_mcp.parser import extract_last_json_array, extract_last_json_object
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, ExecutionMode, PromptRequest


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
            agent=self.AGENT_NAME,
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
            return

        error = data.get("error")
        if not isinstance(error, dict):
            return

        code = self._coerce_error_code(error.get("code", "unknown"))
        message = error.get("message", "unknown error")
        status = error.get("status", "")
        if code == 1 and message == "[object Object]":
            return
        error_msg = f"Gemini API error {code}: {message} ({status})"
        self._raise_structured_error(error_msg, code, stdout, stderr, returncode, command)
