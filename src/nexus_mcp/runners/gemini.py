# src/nexus_mcp/runners/gemini.py
"""Gemini CLI runner implementation.

Executes Google's Gemini CLI (https://github.com/google/generative-ai-python)
with JSON output format.

Command format:
    gemini -p "<prompt>" --output-format json [--model <model>] [--sandbox|--yolo]

Expected JSON response:
    {"response": "...", "stats": {...}}  # stats field is optional
"""

import json

from nexus_mcp.cli_detector import detect_cli, get_cli_capabilities, get_cli_version
from nexus_mcp.config import get_agent_env
from nexus_mcp.exceptions import CLINotFoundError, ParseError
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
            stdout: JSON output from Gemini CLI.
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
        """
        # Parse JSON
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as e:
            raise ParseError(
                f"Invalid JSON from Gemini CLI: {e}",
                raw_output=stdout,
            ) from None

        # Validate 'response' field exists
        if "response" not in data:
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

    def _recover_from_error(
        self, stdout: str, stderr: str, returncode: int
    ) -> AgentResponse | None:
        """Attempt to parse stdout even on non-zero exit code.

        Gemini CLI sometimes produces valid JSON in stdout even when
        encountering errors (e.g., rate limits, warnings).

        Args:
            stdout: Subprocess stdout
            stderr: Subprocess stderr
            returncode: Exit code

        Returns:
            Recovered AgentResponse if stdout is valid JSON, None otherwise
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
        except Exception:
            # Recovery failed - let AbstractRunner raise SubprocessError
            return None
