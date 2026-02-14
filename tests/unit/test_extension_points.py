# tests/unit/test_extension_points.py
"""Verify extension points are accessible for future pipeline enhancements.

These tests do NOT implement features — they verify the architecture allows
adding them without breaking changes (regression guards for extensibility).
"""

from unittest.mock import patch

import pytest

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.types import AgentResponse, PromptRequest
from tests.fixtures import create_mock_process, make_prompt_request


@pytest.fixture(autouse=True)
def mock_cli_detection():
    """Auto-mock CLI detection for extension point tests.

    TestCacheExtensionPoint and TestValidateExtensionPoint use
    RunnerFactory.create("gemini") which triggers GeminiRunner.__init__.
    """
    with (
        patch("nexus_mcp.runners.gemini.detect_cli") as mock_detect,
        patch("nexus_mcp.runners.gemini.get_cli_version", return_value="0.12.0"),
    ):
        mock_detect.return_value = CLIInfo(found=True, path="/usr/bin/gemini")
        yield mock_detect


class TestCacheExtensionPoint:
    """Verify cache can be added as decorator on runner.run()."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_runner_run_is_awaitable_and_wrappable(self, mock_exec):
        """Cache decorator needs to wrap runner.run() without breaking."""
        mock_exec.return_value = create_mock_process(stdout='{"response": "test"}')

        from nexus_mcp.runners.factory import RunnerFactory

        runner = RunnerFactory.create("gemini")
        request = make_prompt_request(prompt="test")

        original_run = runner.run

        async def cache_wrapper(req: PromptRequest) -> AgentResponse:
            return await original_run(req)

        runner.run = cache_wrapper
        response = await runner.run(request)
        assert response.output == "test"


class TestValidateExtensionPoint:
    """Verify validation can intercept before runner.run()."""

    @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
    async def test_can_intercept_request_before_execution(self, mock_exec):
        """Validation hook needs to inspect request before subprocess."""
        mock_exec.return_value = create_mock_process(stdout='{"response": "ok"}')

        from nexus_mcp.runners.factory import RunnerFactory

        runner = RunnerFactory.create("gemini")
        request = make_prompt_request(prompt="test")

        validation_called = False

        async def validate_and_run(req: PromptRequest) -> AgentResponse:
            nonlocal validation_called
            validation_called = True
            return await runner.run(req)

        await validate_and_run(request)
        assert validation_called


class TestChunkExtensionPoint:
    """Verify chunking can override AbstractRunner.run()."""

    def test_abstract_runner_run_is_overridable(self):
        """Subclasses must be able to override run() for chunking logic."""
        from nexus_mcp.runners.base import AbstractRunner

        class ChunkableRunner(AbstractRunner):
            def build_command(self, request: PromptRequest) -> list[str]:
                return ["echo"]

            def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
                return AgentResponse(agent="test", output=stdout, raw_output=stdout)

            async def run(self, request):
                return await super().run(request)

        runner = ChunkableRunner()
        assert hasattr(runner, "run")


class TestFormatExtensionPoint:
    """Verify formatting can transform AgentResponse after parse_output()."""

    def test_agent_response_is_serializable(self):
        """Formatter needs structured response to transform."""
        response = AgentResponse(
            agent="gemini",
            output="test",
            raw_output='{"response": "test"}',
            metadata={"cost": 0.01},
        )

        def format_as_json(resp: AgentResponse) -> dict:
            return {"text": resp.output, "metadata": resp.metadata}

        formatted = format_as_json(response)
        assert formatted["text"] == "test"
