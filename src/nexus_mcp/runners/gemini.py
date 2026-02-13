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

from nexus_mcp.exceptions import ParseError
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, PromptRequest


class GeminiRunner(AbstractRunner):
    """Runner for Google Gemini CLI.

    Builds commands in format:
        gemini -p <prompt> --output-format json [options]

    Parses JSON responses:
        {"response": "text", "stats": {...}}
    """

    def build_command(self, request: PromptRequest) -> list[str]:
        """Build Gemini CLI command from request.

        Args:
            request: PromptRequest with prompt, model, execution_mode.

        Returns:
            Command list: ["gemini", "-p", <prompt>, "--output-format", "json", ...]

        Command structure:
            1. Base: gemini -p <prompt> --output-format json
            2. Add --model <model> if request.model is set
            3. Add --sandbox if execution_mode == "sandbox"
            4. Add --yolo if execution_mode == "yolo"
        """
        command = [
            "gemini",
            "-p",
            request.prompt,
            "--output-format",
            "json",
        ]

        # Add model flag if specified
        if request.model:
            command.extend(["--model", request.model])

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
