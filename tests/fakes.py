"""Test-only fake runner for generic Nexus core tests."""

from typing import ClassVar

from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, ExecutionMode, LogEmitter, ProgressEmitter, PromptRequest


class FakeRunner(AbstractRunner):
    """Runner used only by tests where any runner would do."""

    AGENT_NAME = "fake"
    _SUPPORTED_MODES: ClassVar[tuple[ExecutionMode, ...]] = ("default", "yolo")

    def __init__(self) -> None:
        self.timeout = 30
        self.base_delay = 0.01
        self.max_delay = 0.01
        self.default_max_attempts = 1
        self.output_limit = 50_000
        self.default_model = None
        self.cli_path = self.AGENT_NAME

    async def run(
        self,
        request: PromptRequest,
        emitter: LogEmitter | None = None,
        progress: ProgressEmitter | None = None,
    ) -> AgentResponse:
        # Intentionally bypasses AbstractRunner's subprocess pipeline for generic tests.
        output = request.context.get("fake_output", "fake output")
        raw_output = str(output)
        return AgentResponse(cli=self.AGENT_NAME, output=raw_output, raw_output=raw_output)

    def build_command(self, request: PromptRequest) -> list[str]:
        return [self.cli_path, request.prompt]

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        return AgentResponse(cli=self.AGENT_NAME, output=stdout, raw_output=stdout)
