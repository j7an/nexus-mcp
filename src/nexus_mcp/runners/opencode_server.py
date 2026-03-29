"""OpenCode Server HTTP runner implementation.

Communicates with `opencode serve` over HTTP + SSE instead of spawning
CLI subprocesses. Parallel runner alongside the existing OpenCodeRunner (CLI).

Overrides _execute() directly — no build_command() or parse_output() since
there's no subprocess to build/parse.
"""

from typing import ClassVar

from nexus_mcp.config_resolver import get_runner_defaults
from nexus_mcp.exceptions import RetryableError
from nexus_mcp.http_client import get_http_client
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, ExecutionMode, LogEmitter, ProgressEmitter, PromptRequest


class OpenCodeServerRunner(AbstractRunner):
    """Runner that communicates with opencode serve over HTTP.

    Unlike other runners, this does not spawn a subprocess. It overrides
    _execute() to use OpenCodeHTTPClient for the entire prompt flow:
    health check → session → SSE subscribe → prompt → collect response.
    """

    AGENT_NAME = "opencode_server"
    _SUPPORTED_MODES: ClassVar[tuple[ExecutionMode, ...]] = ("default",)

    def __init__(self) -> None:
        """Initialize runner with HTTP-specific config.

        Skips binary detection — opencode_server connects to a remote server.
        CLI detection is special-cased in cli_detector.py.
        """
        defaults = get_runner_defaults(self.AGENT_NAME)
        self.timeout: int = defaults.timeout  # type: ignore[assignment]
        self.base_delay: float = defaults.retry_base_delay  # type: ignore[assignment]
        self.max_delay: float = defaults.retry_max_delay  # type: ignore[assignment]
        self.default_max_attempts: int = defaults.max_retries  # type: ignore[assignment]
        self.output_limit: int = defaults.output_limit  # type: ignore[assignment]
        self.default_model: str | None = defaults.model
        self.cli_path: str = "http"
        self._client = get_http_client()

    async def _execute(
        self, request: PromptRequest, emit: LogEmitter, progress: ProgressEmitter
    ) -> AgentResponse:
        """Execute prompt via HTTP flow.

        Steps:
        1. Health check (GET /global/health)
        2. Session resolve (create or reuse via label)
        3. SSE subscribe + send prompt (race condition mitigation)
        4. Collect response text
        5. Return AgentResponse
        """
        # Step 1: Health check
        await progress(1, 5, "Checking server health")
        healthy = await self._client.health_check()
        if not healthy:
            raise RetryableError("opencode serve not reachable at configured URL")

        # Step 2: Session resolve
        await progress(2, 5, "Resolving session")
        label = request.context.get("label") if request.context else None
        session_id = await self._client.resolve_session(label)
        await emit("info", f"Using session {session_id}")

        # Step 3+4: SSE subscribe + prompt + collect
        await progress(3, 5, "Sending prompt via SSE")
        prompt_text = self._build_prompt(request)
        output = await self._client.send_prompt(session_id, prompt_text)

        # Step 5: Build response
        await progress(5, 5, "Building response")
        response = AgentResponse(
            cli=self.AGENT_NAME,
            output=output,
            raw_output=output,
        )
        return self._apply_output_limit(response, request)

    def build_command(self, request: PromptRequest) -> list[str]:
        """Not applicable — HTTP runner has no CLI command."""
        raise NotImplementedError("OpenCodeServerRunner uses HTTP, not CLI subprocess")

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        """Not applicable — HTTP runner has no stdout/stderr to parse."""
        raise NotImplementedError("OpenCodeServerRunner uses HTTP, not CLI subprocess")
