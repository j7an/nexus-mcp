# tests/unit/runners/test_gemini.py
"""Tests for GeminiRunner.

Tests verify:
- Command building: gemini -p <prompt> --output-format json [--model X] [--sandbox/--yolo]
- Output parsing: {"response": "...", "stats": {...}} → AgentResponse
- Error handling: invalid JSON, missing fields, subprocess errors
"""

from unittest.mock import patch

import pytest

from nexus_mcp.exceptions import ParseError, SubprocessError
from nexus_mcp.runners.gemini import GeminiRunner
from tests.fixtures import (
    GEMINI_JSON_RESPONSE,
    GEMINI_JSON_WITH_STATS,
    create_mock_process,
    make_prompt_request,
)


class TestGeminiRunnerCommandBuilding:
    """Test GeminiRunner.build_command() constructs correct CLI commands."""

    def test_build_command_default_mode(self):
        """Default execution mode should produce: gemini -p <prompt> --output-format json."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="Hello, Gemini!")

        command = runner.build_command(request)

        assert command == ["gemini", "-p", "Hello, Gemini!", "--output-format", "json"]

    def test_build_command_with_model(self):
        """Custom model should add --model flag."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="test", model="gemini-2.5-pro")

        command = runner.build_command(request)

        assert command == [
            "gemini",
            "-p",
            "test",
            "--output-format",
            "json",
            "--model",
            "gemini-2.5-pro",
        ]

    def test_build_command_sandbox_mode(self):
        """Sandbox mode should add --sandbox flag."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="test", execution_mode="sandbox")

        command = runner.build_command(request)

        assert command == [
            "gemini",
            "-p",
            "test",
            "--output-format",
            "json",
            "--sandbox",
        ]

    def test_build_command_yolo_mode(self):
        """YOLO mode should add --yolo flag."""
        runner = GeminiRunner()
        request = make_prompt_request(prompt="test", execution_mode="yolo")

        command = runner.build_command(request)

        assert command == [
            "gemini",
            "-p",
            "test",
            "--output-format",
            "json",
            "--yolo",
        ]

    def test_build_command_all_options(self):
        """All options together: model + yolo mode."""
        runner = GeminiRunner()
        request = make_prompt_request(
            prompt="complex query",
            model="gemini-2.5-flash",
            execution_mode="yolo",
        )

        command = runner.build_command(request)

        assert command == [
            "gemini",
            "-p",
            "complex query",
            "--output-format",
            "json",
            "--model",
            "gemini-2.5-flash",
            "--yolo",
        ]


class TestGeminiRunnerOutputParsing:
    """Test GeminiRunner.parse_output() handles Gemini CLI JSON responses."""

    def test_parse_output_minimal_json(self):
        """Minimal valid JSON: {"response": "text"} → AgentResponse."""
        runner = GeminiRunner()

        response = runner.parse_output(GEMINI_JSON_RESPONSE, stderr="")

        assert response.agent == "gemini"
        assert response.output == "test output"
        assert response.raw_output == GEMINI_JSON_RESPONSE
        assert response.metadata == {}

    def test_parse_output_with_stats(self):
        """JSON with stats: {"response": "...", "stats": {...}} → metadata."""
        runner = GeminiRunner()

        response = runner.parse_output(GEMINI_JSON_WITH_STATS, stderr="")

        assert response.agent == "gemini"
        assert response.output == "Hello, world!"
        assert response.raw_output == GEMINI_JSON_WITH_STATS
        assert response.metadata == {"stats": {"models": {"gemini-2.5-flash": 1}}}

    def test_parse_output_strips_whitespace(self):
        """Response text should be stripped of leading/trailing whitespace."""
        runner = GeminiRunner()
        json_with_whitespace = '{"response": "  trimmed  "}'

        response = runner.parse_output(json_with_whitespace, stderr="")

        assert response.output == "trimmed"

    def test_parse_output_preserves_internal_whitespace(self):
        """Internal whitespace and newlines should be preserved."""
        runner = GeminiRunner()
        json_with_newlines = '{"response": "Line 1\\n\\nLine 2"}'

        response = runner.parse_output(json_with_newlines, stderr="")

        assert response.output == "Line 1\n\nLine 2"


class TestGeminiRunnerErrorHandling:
    """Test GeminiRunner error handling for invalid/malformed output."""

    def test_parse_output_invalid_json(self):
        """Invalid JSON should raise ParseError with context."""
        runner = GeminiRunner()
        invalid_json = "not valid json {{"

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(invalid_json, stderr="")

        assert "Invalid JSON" in str(exc_info.value)
        assert exc_info.value.raw_output == invalid_json

    def test_parse_output_missing_response_field(self):
        """Missing 'response' field should raise ParseError."""
        runner = GeminiRunner()
        missing_field = '{"stats": {}}'  # No "response" key

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(missing_field, stderr="")

        assert "Missing 'response' field" in str(exc_info.value)
        assert exc_info.value.raw_output == missing_field

    def test_parse_output_non_string_response(self):
        """Non-string 'response' value should raise ParseError."""
        runner = GeminiRunner()
        non_string_response = '{"response": 123}'  # response should be string

        with pytest.raises(ParseError) as exc_info:
            runner.parse_output(non_string_response, stderr="")

        assert "'response' field must be a string" in str(exc_info.value)
        assert exc_info.value.raw_output == non_string_response


class TestGeminiRunnerIntegration:
    """Test GeminiRunner.run() end-to-end with mocked subprocess."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_success_returns_parsed_response(self, mock_exec):
        """Successful run should execute CLI and return parsed AgentResponse."""
        # Arrange
        mock_exec.return_value = create_mock_process(
            stdout=GEMINI_JSON_WITH_STATS,
            returncode=0,
        )
        runner = GeminiRunner()
        request = make_prompt_request(
            agent="gemini",
            prompt="test prompt",
            model="gemini-2.5-flash",
        )

        # Act
        response = await runner.run(request)

        # Assert: Command built correctly
        mock_exec.assert_awaited_once_with(
            "gemini",
            "-p",
            "test prompt",
            "--output-format",
            "json",
            "--model",
            "gemini-2.5-flash",
            stdout=-1,
            stderr=-1,
        )

        # Assert: Response parsed correctly
        assert response.agent == "gemini"
        assert response.output == "Hello, world!"
        assert response.metadata["stats"]["models"]["gemini-2.5-flash"] == 1

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_run_subprocess_error_propagates(self, mock_exec):
        """Subprocess errors (non-zero exit) should raise SubprocessError."""
        # Arrange: CLI exits with error
        mock_exec.return_value = create_mock_process(
            stdout="",
            stderr="API key not found",
            returncode=1,
        )
        runner = GeminiRunner()
        request = make_prompt_request()

        # Act & Assert
        with pytest.raises(SubprocessError) as exc_info:
            await runner.run(request)

        assert exc_info.value.returncode == 1
        assert "API key not found" in exc_info.value.stderr
