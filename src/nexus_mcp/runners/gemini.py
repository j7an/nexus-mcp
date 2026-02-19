# src/nexus_mcp/runners/gemini.py
"""Gemini CLI runner implementation.

Executes Google's Gemini CLI (https://github.com/google/generative-ai-python)
with JSON output format.

Command format:
    gemini -p "<prompt>" --output-format json [--model <model>] [--sandbox|--yolo]

Expected JSON response:
    {"response": "...", "stats": {...}}  # stats field is optional
"""

import contextlib
import json
from typing import Any

from nexus_mcp.cli_detector import detect_cli, get_cli_capabilities, get_cli_version
from nexus_mcp.config import get_agent_env
from nexus_mcp.exceptions import CLINotFoundError, ParseError, SubprocessError
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, PromptRequest


class GeminiRunner(AbstractRunner):
    """Runner for Google Gemini CLI.

    Builds commands in format:
        gemini -p <prompt> --output-format json [options]

    Parses JSON responses:
        {"response": "text", "stats": {...}}
    """

    def __init__(self) -> None:
        """Initialize GeminiRunner with CLI detection and env configuration.

        Raises:
            CLINotFoundError: If gemini CLI is not found in PATH.
        """
        # Phase 3: CLI detection
        info = detect_cli("gemini")
        if not info.found:
            raise CLINotFoundError("gemini")
        version = get_cli_version("gemini")
        self.capabilities = get_cli_capabilities("gemini", version)

        # Phase 3.7: env config layer
        super().__init__()  # sets self.timeout from AbstractRunner
        self.cli_path: str = get_agent_env("gemini", "PATH", default="gemini") or "gemini"
        self.default_model: str | None = get_agent_env("gemini", "MODEL")

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
            4. Add --sandbox if execution_mode == "sandbox"
            5. Add --yolo if execution_mode == "yolo"
        """
        # Build prompt with file references if provided
        prompt = request.prompt
        if request.file_refs:
            file_list = "\n".join(f"- {path}" for path in request.file_refs)
            prompt = f"{prompt}\n\nFile references:\n{file_list}"

        # Use configured CLI path
        command = [self.cli_path, "-p", prompt]

        # Add --output-format json if supported by CLI version
        if self.capabilities.supports_json:
            command.extend(["--output-format", "json"])

        # Determine model: request.model > env default > CLI default
        model = request.model or self.default_model
        if model:
            command.extend(["--model", model])

        # Add execution mode flag
        match request.execution_mode:
            case "sandbox":
                command.append("--sandbox")
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
            data = self._extract_last_json_object(stdout)
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
        metadata = data.get("stats", {})

        return AgentResponse(
            agent="gemini",
            output=response_text.strip(),
            raw_output=stdout,
            metadata=metadata,
        )

    @staticmethod
    def _extract_last_json_object(text: str) -> dict[str, Any] | None:
        """Find and parse the last JSON object in a multi-line string.

        Uses brace-depth matching (no regex) to locate the last complete {...}
        block. Handles Gemini CLI's pattern of appending a JSON error block at
        the end of stderr that may contain log lines and stack traces.

        Args:
            text: String that may contain JSON, possibly mixed with other content.

        Returns:
            Parsed dict if a valid JSON object is found, None otherwise.
        """
        if not text:
            return None

        last_close = text.rfind("}")
        if last_close == -1:
            return None

        depth = 0
        for i in range(last_close, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[i : last_close + 1])
                        return parsed if isinstance(parsed, dict) else None
                    except (json.JSONDecodeError, ValueError, RecursionError):
                        return None

        return None

    @staticmethod
    def _extract_last_json_array(text: str) -> dict[str, Any] | None:
        """Find and parse the last JSON array in a multi-line string, returning its first element.

        Scans rightward-to-left through ']' positions using bracket-depth matching.
        Handles Gemini CLI's GaxiosError format where errors are wrapped in an array:
        [{"error": {...}}]. Skips over ']' characters that appear inside string values
        (e.g. "[object Object]" in the session summary) by trying each candidate in turn.

        Args:
            text: String that may contain a JSON array, possibly mixed with other content.

        Returns:
            First element of the parsed array if it's a dict, None otherwise.
        """
        if not text:
            return None

        search_end = len(text)
        while True:
            last_close = text.rfind("]", 0, search_end)
            if last_close == -1:
                return None

            depth = 0
            start = -1
            for i in range(last_close, -1, -1):
                if text[i] == "]":
                    depth += 1
                elif text[i] == "[":
                    depth -= 1
                    if depth == 0:
                        start = i
                        break

            if start != -1:
                with contextlib.suppress(json.JSONDecodeError, ValueError, RecursionError):
                    parsed = json.loads(text[start : last_close + 1])
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                        return parsed[0]

            search_end = last_close

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
            data = self._extract_last_json_array(stderr)
        if data is None:
            data = self._extract_last_json_object(stderr)

        if data is None:
            return

        error = data.get("error")
        if not isinstance(error, dict):
            return

        code = error.get("code", "unknown")
        message = error.get("message", "unknown error")
        status = error.get("status", "")
        if code == 1 and message == "[object Object]":
            return
        raise SubprocessError(
            f"Gemini API error {code}: {message} ({status})",
            stderr=stderr,
            stdout=stdout,
            returncode=returncode,
            command=command,
        )

    def _recover_from_error(
        self, stdout: str, stderr: str, returncode: int, command: list[str] | None = None
    ) -> AgentResponse | None:
        """Attempt to parse stdout even on non-zero exit code.

        Gemini CLI sometimes produces valid JSON in stdout even when
        encountering errors (e.g., rate limits, warnings). If stdout contains
        a structured API error object, raises a descriptive SubprocessError.

        Args:
            stdout: Subprocess stdout
            stderr: Subprocess stderr
            returncode: Exit code
            command: The CLI command that was run (forwarded to SubprocessError if raised).

        Returns:
            Recovered AgentResponse if stdout is valid JSON, None otherwise

        Raises:
            SubprocessError: If stdout or stderr contains a Gemini API error object.
        """
        try:
            response = self.parse_output(stdout, stderr)
            # Mark as recovered in metadata
            metadata = response.metadata.copy()
            metadata["recovered_from_error"] = True
            metadata["original_exit_code"] = returncode
            metadata["stderr"] = stderr

            return AgentResponse(
                agent=response.agent,
                output=response.output,
                raw_output=response.raw_output,
                metadata=metadata,
            )
        except ParseError:
            # Try to extract structured API error; raises SubprocessError if found
            self._try_extract_error(stdout, stderr, returncode, command)
            return None  # Falls through to base.py's generic SubprocessError
