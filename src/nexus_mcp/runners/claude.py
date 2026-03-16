# src/nexus_mcp/runners/claude.py
"""Claude Code CLI runner implementation.

Executes Anthropic's Claude Code CLI with JSON output format.

Command format:
    claude -p "<prompt>" --output-format json [--model <model>]
    [--dangerously-skip-permissions]

Expected JSON response (array or single object):
    [
      {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "..."}]},
        "cost_usd": 0.001,
        "duration_ms": 1234
      },
      {
        "type": "result",
        "result": "response text",
        "session_id": "sess-abc123",
        "cost_usd": 0.005,          # older CLI versions
        "total_cost_usd": 0.005,    # newer CLI versions (normalized → cost_usd)
        "duration_ms": 5678,
        "num_turns": 1
      }
    ]

Note: Claude Code has no sandbox concept. Only "default" and "yolo" modes are supported.
"""

import json
from typing import Any

from nexus_mcp.exceptions import ParseError
from nexus_mcp.parser import extract_last_json_list, extract_last_json_object
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, PromptRequest


class ClaudeRunner(AbstractRunner):
    """Runner for Claude Code CLI.

    Builds commands in format:
        claude -p <prompt> --output-format json [options]

    Parses JSON array responses, extracting text from the result element.
    """

    AGENT_NAME = "claude"

    def build_command(self, request: PromptRequest) -> list[str]:
        """Build Claude Code CLI command from request.

        Args:
            request: PromptRequest with prompt, model, execution_mode, file_refs.

        Returns:
            Command list: [cli_path, "-p", <prompt>, "--output-format", "json", ...]

        Command structure:
            1. Base: {cli_path} -p <prompt> --output-format json
            2. Add --model <model> (request.model > env default > CLI default)
            3. Add --dangerously-skip-permissions if execution_mode == "yolo"
        """
        command = [self.cli_path, "-p", self._build_prompt(request), "--output-format", "json"]

        model = request.model or self.default_model
        if model:
            command.extend(["--model", model])

        match request.execution_mode:
            case "yolo":
                command.append("--dangerously-skip-permissions")
            case _:
                pass  # default: no extra flags

        return command

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        """Parse Claude Code CLI JSON array output into AgentResponse.

        Args:
            stdout: JSON array output from Claude Code CLI.
            stderr: Standard error output (not used for parsing).

        Returns:
            AgentResponse with:
                - agent: "claude"
                - output: Stripped result text
                - raw_output: Original stdout
                - metadata: cost_usd, duration_ms, num_turns, session_id (when present)

        Raises:
            ParseError: If output is not a valid JSON array, contains no result
                        or assistant elements, or the result field is not a string.
        """
        # Fast path: clean JSON
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Fallback: extract array from noisy stdout (log lines before JSON)
            data = extract_last_json_list(stdout)
            if data is None:
                # Also try single-object fallback for noisy stdout
                obj = extract_last_json_object(stdout)
                if obj is not None:
                    data = [obj]
                else:
                    raise ParseError("Invalid JSON from Claude CLI", raw_output=stdout) from None

        # Newer Claude CLI versions return a single object instead of an array
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ParseError("Expected JSON object or array from Claude CLI", raw_output=stdout)

        # Find last "result" element (primary extraction path)
        result_text, metadata = self._extract_result(data, stdout)
        if result_text is None:
            # Fallback: extract from last assistant message content blocks
            result_text, metadata = self._extract_assistant_text(data)

        if result_text is None:
            raise ParseError("No result or assistant text in Claude CLI output", raw_output=stdout)

        return AgentResponse(
            agent=self.AGENT_NAME,
            output=result_text.strip(),
            raw_output=stdout,
            metadata=metadata,
        )

    @staticmethod
    def _extract_metadata(element: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        """Extract specified metadata keys from element dict.

        Args:
            element: Source dict to extract from.
            keys: Tuple of key names to extract if present.

        Returns:
            Dict containing only the keys that exist in element.
        """
        return {key: element[key] for key in keys if key in element}

    def _extract_result(
        self, data: list[Any], raw_output: str = ""
    ) -> tuple[str | None, dict[str, Any]]:
        """Extract text and metadata from the last 'result' element.

        Checks is_error to surface error responses rather than treating
        them as successful output.

        Args:
            data: Parsed JSON array from Claude CLI.
            raw_output: Original stdout for ParseError context.

        Returns:
            (result_text, metadata) tuple. result_text is None if no result element found.

        Raises:
            ParseError: If result element found but result field is not a string,
                        or if is_error is True (error response from Claude).
        """
        for element in reversed(data):
            if not isinstance(element, dict) or element.get("type") != "result":
                continue
            if element.get("is_error"):
                error_msg = element.get("result") or "unknown error"
                raise ParseError(
                    f"Claude CLI returned an error result: {error_msg}",
                    raw_output=raw_output,
                )
            result = element.get("result")
            if not isinstance(result, str):
                raise ParseError(
                    f"Expected string 'result', got {type(result).__name__}",
                    raw_output=str(element),
                )
            metadata = self._extract_metadata(
                element, ("cost_usd", "duration_ms", "num_turns", "session_id")
            )
            # Normalize total_cost_usd → cost_usd for newer CLI versions
            if "cost_usd" not in metadata and "total_cost_usd" in element:
                metadata["cost_usd"] = element["total_cost_usd"]
            return result, metadata
        return None, {}

    def _extract_assistant_text(self, data: list[Any]) -> tuple[str | None, dict[str, Any]]:
        """Fallback: extract text and metadata from the last assistant message content blocks.

        Args:
            data: Parsed JSON array from Claude CLI.

        Returns:
            (text, metadata) tuple. text is None if no assistant element with content found.
        """
        for element in reversed(data):
            if not isinstance(element, dict) or element.get("type") != "assistant":
                continue
            message = element.get("message", {})
            content = message.get("content", []) if isinstance(message, dict) else []
            parts: list[str] = []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
            if parts:
                metadata = self._extract_metadata(element, ("cost_usd", "duration_ms"))
                return "\n\n".join(parts), metadata
        return None, {}

    def _try_extract_error(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
        command: list[str] | None = None,
    ) -> None:
        """Extract and raise a structured API error from stderr or stdout JSON.

        Args:
            stdout: Subprocess stdout to inspect as fallback.
            stderr: Subprocess stderr inspected first.
            returncode: Exit code (passed through to SubprocessError).
            command: The CLI command that was run (passed through to SubprocessError).

        Raises:
            RetryableError: If error code is 429 or 503.
            SubprocessError: If stderr or stdout contains a non-retryable error object.
        """
        data = None
        for source in (stderr, stdout):
            candidate = extract_last_json_object(source)
            if candidate and isinstance(candidate.get("error"), dict):
                data = candidate
                break
        if data is None:
            return

        error = data["error"]

        code = self._coerce_error_code(error.get("code", "unknown"))
        message = error.get("message", "unknown error")
        error_msg = f"Claude API error {code}: {message}"
        self._raise_structured_error(error_msg, code, stdout, stderr, returncode, command)
