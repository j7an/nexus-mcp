# src/nexus_mcp/runners/opencode.py
"""OpenCode CLI runner implementation.

Executes OpenCode CLI with JSON output format.

Command format:
    opencode run "<prompt>" --format json [--model provider/model]

The `run` subcommand accepts the prompt as a positional argument.
--format json emits NDJSON events to stdout.

Execution modes:
    OpenCode's `run` subcommand does not expose tool restriction flags.
    All execution modes produce the same command structure.

Output format (NDJSON events, verified against OpenCode CLI v1.2.20):
    Primary:  {"type": "text", "part": {"type": "text", "text": "<output>"}}
    Errors:   {"type": "error", "error": {"name": "...", "data": {"message": "..."}}}
    Lifecycle: step_start, step_finish, tool_use events (ignored by parser)
    Fallback: single JSON object with message/content/text/response key
"""

import json
from typing import ClassVar

from nexus_mcp.exceptions import ParseError
from nexus_mcp.parser import extract_last_json_object
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, ExecutionMode, PromptRequest

_JSON_OUTPUT_KEYS = ("message", "content", "text", "response")


class OpenCodeRunner(AbstractRunner):
    """Runner for OpenCode CLI.

    Builds commands in format:
        opencode run <prompt> --format json [--model <model>]
    """

    AGENT_NAME = "opencode"
    _SUPPORTED_MODES: ClassVar[tuple[ExecutionMode, ...]] = ("default",)

    def build_command(self, request: PromptRequest) -> list[str]:
        """Build OpenCode CLI command from request.

        Args:
            request: PromptRequest with prompt, model, execution_mode, file_refs.

        Returns:
            Command list: [cli_path, "run", <prompt>, "--format", "json", ...]

        Command structure:
            1. Base: {cli_path} run <prompt> --format json
            2. Add --model <model> (request.model > env default > CLI default)
            3. Execution mode: no tool restriction flags available in opencode run
        """
        command = [self.cli_path, "run", self._build_prompt(request), "--format", "json"]

        model = request.model or self.default_model
        if model:
            command.extend(["--model", model])

        return command

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:  # noqa: ARG002
        """Parse OpenCode CLI output into AgentResponse.

        Tries two parse paths in order:
        1. NDJSON events (OpenCode schema: type=text, part.text)
        2. Single JSON object with message/content/text/response key

        Args:
            stdout: CLI output to parse.
            stderr: Standard error output (not used for parsing).

        Returns:
            AgentResponse with cli="opencode", parsed output, raw_output=stdout.

        Raises:
            ParseError: If neither parse path yields output.
                Note: valid NDJSON with no text events returns output="" (tool-only runs).
        """
        # Primary: NDJSON events (OpenCode schema)
        ndjson_output = self._parse_opencode_ndjson(stdout)
        if ndjson_output is not None:
            return AgentResponse(
                cli=self.AGENT_NAME,
                output=ndjson_output.strip(),
                raw_output=stdout,
            )

        # Fallback: single JSON object
        json_output = self._parse_json_object(stdout)
        if json_output is not None:
            return AgentResponse(
                cli=self.AGENT_NAME,
                output=json_output.strip(),
                raw_output=stdout,
            )

        raise ParseError(
            "No parseable output found in OpenCode CLI response",
            raw_output=stdout,
        )

    def _parse_opencode_ndjson(self, stdout: str) -> str | None:
        """Extract text from OpenCode NDJSON event stream.

        OpenCode emits one JSON event per line. Text events have:
            {"type": "text", "part": {"type": "text", "text": "<content>"}}

        Error events ({"type": "error", ...}) are skipped so that error-only
        NDJSON falls through to _try_extract_error rather than returning empty
        output and silently losing the structured error.

        Collects text from all ``type=text`` events and joins with ``"\\n\\n"``.

        Args:
            stdout: Raw NDJSON output from OpenCode CLI.

        Returns:
            Joined text from all text events, ``""`` if valid NDJSON was parsed
            but contained no text events (e.g. tool-only runs), or ``None`` if
            stdout is not NDJSON at all (triggers JSON fallback).
        """
        parts: list[str] = []
        found_json_event = False
        for line in stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                continue  # Skipped; _try_extract_error handles these on non-zero exit
            if "type" not in event:
                continue  # Typeless dicts: skip sentinel, let JSON fallback handle them
            found_json_event = True
            if event.get("type") != "text":
                continue
            part = event.get("part")
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n\n".join(parts)
        return "" if found_json_event else None

    def _parse_json_object(self, stdout: str) -> str | None:
        """Extract output text from a single JSON object response.

        Tries json.loads first for clean output, falls back to
        extract_last_json_object for noisy stdout (log lines before JSON).
        Non-dict JSON (arrays, scalars) also falls through to extract_last_json_object.
        Extracts the first non-empty string value from _JSON_OUTPUT_KEYS.

        Args:
            stdout: Raw CLI output to parse.

        Returns:
            Extracted text string, or None if no matching key found.
        """
        data: dict[str, object] | None = None
        try:
            parsed = json.loads(stdout)
            data = parsed if isinstance(parsed, dict) else extract_last_json_object(stdout)
        except json.JSONDecodeError:
            data = extract_last_json_object(stdout)

        if data is None:
            return None

        for key in _JSON_OUTPUT_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value:
                return value

        return None

    def _try_extract_error(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        command: list[str] | None = None,
    ) -> None:
        """Extract and raise a structured API error from OpenCode output.

        Checks two formats in order:
        1. NDJSON error event in stdout: {"type":"error","error":{"name":"...","data":{...}}}
        2. Legacy JSON in stderr/stdout: {"error":{"code":...,"message":"..."}}

        Args:
            stdout: Subprocess stdout; scanned for NDJSON error events first.
            stderr: Subprocess stderr; checked for legacy error JSON.
            returncode: Exit code (passed through to SubprocessError).
            command: The CLI command that was run (passed through to SubprocessError).

        Raises:
            RetryableError: If error code/statusCode is 429 or 503.
            SubprocessError: If a non-retryable structured error is found.
        """
        # 1. Scan stdout NDJSON for error events
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "error":
                continue
            error = event.get("error")
            if not isinstance(error, dict):
                continue

            name = error.get("name", "unknown")
            error_data = error.get("data", {})
            if not isinstance(error_data, dict):
                error_data = {}
            message = error_data.get("message", "unknown error")

            # Use statusCode for retryability if present, else error name
            code = self._coerce_error_code(error_data.get("statusCode", name))

            error_msg = f"OpenCode API error {name}: {message}"
            self._raise_structured_error(error_msg, code, stdout, stderr, returncode, command)

        # 2. Fallback: {"error": {"code": ..., "message": ...}} in stderr/stdout
        legacy: dict[str, object] | None = None
        for source in (stderr, stdout):
            candidate = extract_last_json_object(source)
            if candidate and isinstance(candidate.get("error"), dict):
                legacy = candidate
                break
        if legacy is None:
            return

        error = legacy["error"]
        if not isinstance(error, dict):
            return
        code = self._coerce_error_code(error.get("code", "unknown"))
        message = error.get("message", "unknown error")
        error_msg = f"OpenCode API error {code}: {message}"
        self._raise_structured_error(error_msg, code, stdout, stderr, returncode, command)
