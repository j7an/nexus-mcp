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

    async def _resolve_session(self, label: str | None) -> str:
        """Resolve a session ID for the given label.

        Args:
            label: Task label for session reuse. None = ephemeral session.

        Returns:
            OpenCode session ID string.
        """
        if label is not None and label in self._sessions:
            session_id = self._sessions[label]
            if await self._validate_session(session_id):
                return session_id
            del self._sessions[label]

        session_id = await self._create_session(label)
        if label is not None:
            self._sessions[label] = session_id
        return session_id

    async def _create_session(self, title: str | None) -> str:
        """Create a new OpenCode session via POST /session.

        Args:
            title: Optional session title.

        Returns:
            New session ID.
        """
        body: dict[str, str] = {}
        if title is not None:
            body["title"] = title
        resp = await self._client.post("/session", json=body)
        resp.raise_for_status()
        return str(resp.json()["id"])

    async def _validate_session(self, session_id: str) -> bool:
        """Check if a session still exists via GET /session/:id.

        Returns:
            True if session exists (200), False if gone (404).
        """
        resp = await self._client.get(f"/session/{session_id}")
        return resp.status_code == 200
