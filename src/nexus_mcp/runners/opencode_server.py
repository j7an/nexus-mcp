# src/nexus_mcp/runners/opencode_server.py
"""OpenCode server-mode runner.

Communicates with OpenCode's HTTP API instead of spawning subprocesses.
Uses async prompting (POST /session/:id/prompt_async) with SSE event streaming.

See: docs/superpowers/specs/2026-03-18-opencode-server-runner-design.md
"""

import logging
from typing import ClassVar

import httpx

from nexus_mcp.config import (
    get_opencode_server_password,
    get_opencode_server_url,
    get_opencode_server_username,
)
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.types import AgentResponse, ExecutionMode, PromptRequest

logger = logging.getLogger(__name__)


class OpenCodeServerRunner(AbstractRunner):
    """Runner for OpenCode HTTP server mode.

    Overrides _execute() to communicate via HTTP instead of subprocesses.
    build_command() and parse_output() satisfy the ABC but are never called.
    """

    AGENT_NAME = "opencode_server"
    _SUPPORTED_MODES: ClassVar[tuple[ExecutionMode, ...]] = ("default",)

    def __init__(self) -> None:
        super().__init__()
        self._base_url = get_opencode_server_url()
        password = get_opencode_server_password()
        auth: httpx.BasicAuth | None = None
        if password is not None:
            auth = httpx.BasicAuth(get_opencode_server_username(), password)
        self._client = httpx.AsyncClient(base_url=self._base_url, auth=auth)
        self._sessions: dict[str, str] = {}  # label → OpenCode session ID

    def build_command(self, request: PromptRequest) -> list[str]:
        raise NotImplementedError("OpenCodeServerRunner uses HTTP, not CLI commands")

    def parse_output(self, stdout: str, stderr: str) -> AgentResponse:
        raise NotImplementedError("OpenCodeServerRunner uses HTTP, not CLI output parsing")
