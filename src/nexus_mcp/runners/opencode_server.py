# src/nexus_mcp/runners/opencode_server.py
"""OpenCode server-mode runner.

Communicates with OpenCode's HTTP API instead of spawning subprocesses.
Uses async prompting (POST /session/:id/prompt_async) with SSE event streaming.

See: docs/superpowers/specs/2026-03-18-opencode-server-runner-design.md
"""

import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import ClassVar

import httpx

from nexus_mcp.config import (
    get_opencode_server_password,
    get_opencode_server_url,
    get_opencode_server_username,
)
from nexus_mcp.exceptions import RetryableError, SubprocessError
from nexus_mcp.runners.base import AbstractRunner
from nexus_mcp.sse import parse_sse_stream
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

    async def _execute(self, request: PromptRequest) -> AgentResponse:
        """Execute prompt via OpenCode HTTP API.

        Overrides AbstractRunner._execute() entirely — no subprocess involved.

        Steps:
        1. Resolve session (reuse or create)
        2. POST /session/:id/prompt_async
        3. Consume SSE from GET /global/event
        4. Build AgentResponse
        5. Apply output limit (inherited)
        """
        label = request.context.get("_nexus_label")
        session_id = await self._resolve_session(label)

        # Build payload
        payload: dict[str, object] = {
            "parts": [{"type": "text", "text": self._build_prompt(request)}],
        }
        model = request.model or self.default_model
        if model:
            payload["model"] = model
        if request.context:
            payload["system"] = json.dumps(request.context)

        # Fire async prompt
        effective_timeout = request.timeout if request.timeout is not None else self.timeout
        try:
            resp = await self._client.post(
                f"/session/{session_id}/prompt_async",
                json=payload,
                timeout=float(effective_timeout),
            )
            self._check_http_error(resp)
        except httpx.ConnectError as e:
            raise RetryableError(
                f"Cannot connect to OpenCode server at {self._base_url}: {e}",
                retry_after=None,
            ) from e
        except httpx.TimeoutException as e:
            raise RetryableError(
                f"OpenCode server request timed out: {e}",
                retry_after=None,
            ) from e

        # Consume SSE
        output_parts, raw_events = await self._consume_sse(session_id, effective_timeout)

        response = AgentResponse(
            cli=self.AGENT_NAME,
            output="\n\n".join(output_parts).strip(),
            raw_output=raw_events,
            metadata={"session_id": session_id},
        )
        return self._apply_output_limit(response, request)

    async def _consume_sse(self, session_id: str, timeout: int) -> tuple[list[str], str]:
        """Consume SSE events from GET /global/event for the given session.

        Args:
            session_id: Filter events for this session.
            timeout: HTTP timeout seconds.

        Returns:
            Tuple of (text parts list, raw SSE text).
        """
        parts: list[str] = []
        raw_lines: list[str] = []

        try:
            async with self._client.stream("GET", "/global/event", timeout=float(timeout)) as resp:

                async def line_iter() -> AsyncIterator[str]:
                    async for line in resp.aiter_lines():
                        raw_lines.append(line)
                        yield line

                async for event in parse_sse_stream(line_iter()):
                    if not event.data:
                        continue
                    try:
                        data = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(data, dict):
                        continue

                    event_session = data.get("sessionID")
                    if event_session != session_id:
                        continue

                    event_type = data.get("type")
                    if event_type == "text":
                        part = data.get("part")
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                    elif event_type == "error":
                        self._handle_sse_error(data)
                    elif event_type == "step_finish":
                        break
        except httpx.ConnectError as e:
            raise RetryableError(
                f"SSE connection to OpenCode server failed: {e}",
                retry_after=None,
            ) from e
        except httpx.TimeoutException as e:
            raise RetryableError(
                f"SSE stream timed out: {e}",
                retry_after=None,
            ) from e

        return parts, "\n".join(raw_lines)

    def _check_http_error(self, resp: httpx.Response) -> None:
        """Classify HTTP error responses into retryable/non-retryable.

        Args:
            resp: httpx.Response to check.

        Raises:
            RetryableError: For 429/503 (with Retry-After header if present).
            SubprocessError: For 401/403 and other error status codes.
        """
        if resp.status_code < 400:
            return

        retry_after: float | None = None
        if resp.status_code in (429, 503):
            raw_retry = resp.headers.get("Retry-After")
            if raw_retry is not None:
                with contextlib.suppress(ValueError):
                    retry_after = float(raw_retry)
            raise RetryableError(
                f"OpenCode server returned {resp.status_code}",
                retry_after=retry_after,
                returncode=resp.status_code,
            )

        raise SubprocessError(
            f"OpenCode server returned {resp.status_code}",
            returncode=resp.status_code,
            stderr=resp.text,
        )

    def _handle_sse_error(self, data: dict) -> None:  # type: ignore[type-arg]
        """Handle an SSE error event from the OpenCode stream.

        Args:
            data: Parsed JSON from the SSE data field.

        Raises:
            RetryableError or SubprocessError based on error code.
        """
        error = data.get("error")
        if not isinstance(error, dict):
            return

        name = error.get("name", "unknown")
        error_data = error.get("data", {})
        if not isinstance(error_data, dict):
            error_data = {}
        message = error_data.get("message", "unknown error")
        code = self._coerce_error_code(error_data.get("statusCode", name))
        error_msg = f"OpenCode API error {name}: {message}"
        self._raise_structured_error(error_msg, code, "", "", 1, command=None)
