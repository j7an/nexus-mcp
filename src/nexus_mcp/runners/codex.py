# src/nexus_mcp/runners/codex.py
"""Codex CLI runner implementation.

Executes OpenAI's Codex CLI with NDJSON output format.

Command format:
    codex exec "<prompt>" --json [--model <model>] [--dangerously-bypass-approvals-and-sandbox]

Expected NDJSON response (one JSON object per line):
    {"type": "thread.started", "thread_id": "..."}
    {"type": "item.completed", "item": {"id": "...", "type": "agent_message", "text": "..."}}
    {"type": "turn.completed", "turn_id": "..."}
"""

from typing import ClassVar

from nexus_mcp.exceptions import ParseError
from nexus_mcp.parser import extract_last_json_object, parse_ndjson_events
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, ExecutionMode, PromptRequest


class CodexRunner(AbstractRunner):
    """Runner for OpenAI Codex CLI.

    Builds commands in format:
        codex exec <prompt> --json [options]

    Parses NDJSON responses, collecting text from agent_message events.
    """

    AGENT_NAME = "codex"
    _SUPPORTED_MODES: ClassVar[tuple[ExecutionMode, ...]] = ("default", "yolo")

    def build_command(self, request: PromptRequest) -> list[str]:
        """Build Codex CLI command from request.

        Args:
            request: PromptRequest with prompt, model, execution_mode, file_refs.

        Returns:
            Command list: [cli_path, "exec", <prompt>, "--json", ...]

        Command structure:
            1. Base: {cli_path} exec <prompt> (with file_refs appended if provided)
            2. Add --json
            3. Add --model <model> (request.model > env default > CLI default)
            4. Add --dangerously-bypass-approvals-and-sandbox if execution_mode == "yolo"
        """
        command = [self.cli_path, "exec", self._build_prompt(request), "--json"]

        model = request.model or self.default_model
        if model:
            command.extend(["--model", model])

        match request.execution_mode:
            case "yolo":
                command.append("--dangerously-bypass-approvals-and-sandbox")
            case _:
                pass

        return command

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        """Parse Codex CLI NDJSON output into AgentResponse.

        Args:
            stdout: NDJSON output from Codex CLI (one event per line).
            stderr: Standard error output (not used for parsing).

        Returns:
            AgentResponse with:
                - agent: "codex"
                - output: Stripped joined text from agent_message events
                - raw_output: Original stdout

        Raises:
            ParseError: If no agent_message text can be extracted.
        """
        output = parse_ndjson_events(stdout)
        if output is None:
            raise ParseError(
                "No agent_message text found in Codex CLI NDJSON output",
                raw_output=stdout,
            )
        return AgentResponse(
            agent=self.AGENT_NAME,
            output=output.strip(),
            raw_output=stdout,
        )

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
            SubprocessError: If stderr or stdout contains an error object.
        """
        data = extract_last_json_object(stderr) or extract_last_json_object(stdout)
        if data is None:
            return

        error = data.get("error")
        if not isinstance(error, dict):
            return

        code = self._coerce_error_code(error.get("code", "unknown"))
        message = error.get("message", "unknown error")
        error_msg = f"Codex API error {code}: {message}"
        self._raise_structured_error(error_msg, code, stdout, stderr, returncode, command)
