# src/nexus_mcp/runners/opencode_server.py
"""OpenCode server-mode runner.

Communicates with OpenCode's HTTP API instead of spawning subprocesses.
Uses async prompting (POST /session/:id/prompt_async) with SSE event streaming.

See: docs/superpowers/specs/2026-03-18-opencode-server-runner-design.md
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

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

    async def _check_health(self) -> None:
        """Pre-flight health check via GET /global/health.

        Raises:
            RetryableError: If server is unreachable, unhealthy, or returns an error.
        """
        try:
            resp = await self._client.get("/global/health")
        except httpx.ConnectError as e:
            raise RetryableError(
                f"Cannot connect to OpenCode server at {self._base_url}: {e}",
                retry_after=None,
            ) from e

        if resp.status_code != 200:
            raise RetryableError(
                f"OpenCode server health check failed: HTTP {resp.status_code}",
                retry_after=None,
            )

        try:
            data = resp.json()
        except Exception as e:
            raise RetryableError(
                "OpenCode server health check failed: invalid JSON response",
                retry_after=None,
            ) from e

        if not data.get("healthy"):
            raise RetryableError("OpenCode server reports unhealthy", retry_after=None)

    async def _fetch_session_diff(self, session_id: str) -> list[dict[str, Any]] | None:
        """Fetch file diffs for a session via GET /session/:id/diff.

        Best-effort: returns None on any failure. Never raises.

        Args:
            session_id: The session to fetch diffs for.

        Returns:
            List of diff dicts on success, None on any error.
        """
        try:
            resp = await self._client.get(f"/session/{session_id}/diff")
            if resp.status_code != 200:
                return None
            result: list[dict[str, Any]] = resp.json()
            return result
        except Exception:
            return None

    async def _execute(self, request: PromptRequest) -> AgentResponse:
        """Execute prompt via OpenCode HTTP API.

        Overrides AbstractRunner._execute() entirely — no subprocess involved.

        Steps:
        1. Resolve session (reuse or create)
        2. Open SSE stream (GET /global/event)
        3. POST /session/:id/prompt_async — inside the open stream to avoid race condition
        4. Consume SSE events
        5. Build AgentResponse
        6. Apply output limit (inherited)
        """
        await self._check_health()
        label = request.context.get("_nexus_label")
        session_id = await self._resolve_session(label)

        # Build payload
        payload: dict[str, object] = {
            "parts": [{"type": "text", "text": self._build_prompt(request)}],
        }
        model = request.model or self.default_model
        if model:
            payload["model"] = self._parse_model(model)
        if request.context:
            payload["system"] = json.dumps(request.context)

        effective_timeout = request.timeout if request.timeout is not None else self.timeout

        # SSE stream opens FIRST; prompt fires inside to avoid missing step_finish
        output_parts, raw_events = await self._consume_sse(session_id, payload, effective_timeout)

        response = AgentResponse(
            cli=self.AGENT_NAME,
            output="\n\n".join(output_parts).strip(),
            raw_output=raw_events,
            metadata={"session_id": session_id},
        )
        if not label:
            diff = await self._fetch_session_diff(session_id)
            if diff is not None:
                response = response.with_metadata(diff=diff)

        return self._apply_output_limit(response, request)

    async def _consume_sse(
        self,
        session_id: str,
        payload: dict[str, object],
        timeout: int,
    ) -> tuple[list[str], str]:
        """Open SSE stream, fire prompt_async, then consume events.

        The stream is opened BEFORE posting the prompt to avoid a race condition
        where step_finish is emitted before the client subscribes to /global/event.

        Args:
            session_id: Session to post to and filter events for.
            payload: JSON body for POST /session/:id/prompt_async.
            timeout: HTTP timeout seconds.

        Returns:
            Tuple of (text parts list, raw SSE text).
        """
        text_parts: dict[str, str] = {}  # part_id → final text (deduplicates re-deliveries)
        assistant_msg_ids: set[str] = set()
        raw_lines: list[str] = []

        try:
            async with self._client.stream("GET", "/global/event", timeout=float(timeout)) as resp:
                # Fire prompt AFTER stream is open
                try:
                    post_resp = await self._client.post(
                        f"/session/{session_id}/prompt_async",
                        json=payload,
                        timeout=float(timeout),
                    )
                    self._check_http_error(post_resp)
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

                async def line_iter() -> AsyncGenerator[str]:
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

                    # Real OpenCode schema: {"payload": {"type": ..., "properties": {...}}}
                    envelope = data.get("payload")
                    if not isinstance(envelope, dict):
                        continue
                    event_type = envelope.get("type")
                    props = envelope.get("properties")
                    if not isinstance(props, dict):
                        continue

                    # Session ID path varies by event type:
                    #   message.updated        → props.info.sessionID
                    #   message.part.updated   → props.part.sessionID
                    #   all others             → props.sessionID
                    info_obj = props.get("info")
                    part_obj = props.get("part")
                    if isinstance(info_obj, dict):
                        event_session = info_obj.get("sessionID")
                    elif isinstance(part_obj, dict):
                        event_session = part_obj.get("sessionID")
                    else:
                        event_session = props.get("sessionID")
                    if event_session != session_id:
                        continue

                    if event_type == "message.updated":
                        # Track which message IDs belong to the assistant
                        if isinstance(info_obj, dict) and info_obj.get("role") == "assistant":
                            msg_id = info_obj.get("id")
                            if isinstance(msg_id, str):
                                assistant_msg_ids.add(msg_id)
                    elif event_type == "message.part.updated":
                        if isinstance(part_obj, dict) and part_obj.get("type") == "text":
                            msg_id = part_obj.get("messageID", "")
                            if msg_id in assistant_msg_ids:
                                text = part_obj.get("text")
                                part_id = part_obj.get("id", "")
                                if isinstance(text, str) and text:
                                    text_parts[part_id] = text
                    elif event_type == "error":
                        self._handle_sse_error(props)
                    elif event_type == "session.idle":
                        break
        except httpx.ConnectError as e:
            raise RetryableError(
                f"SSE connection to OpenCode server failed: {e}",
                retry_after=None,
            ) from e
        except httpx.TimeoutException as e:
            await self._abort_session(session_id)
            raise RetryableError(
                f"SSE stream timed out: {e}",
                retry_after=None,
            ) from e
        except asyncio.CancelledError:
            await self._abort_session(session_id)
            raise

        return list(text_parts.values()), "\n".join(raw_lines)

    async def _abort_session(self, session_id: str) -> None:
        """Best-effort abort — POST /session/:id/abort, swallow all errors."""
        try:
            resp = await self._client.post(
                f"/session/{session_id}/abort",
                timeout=5.0,
            )
            logger.info("Abort session %s: HTTP %d", session_id, resp.status_code)
        except Exception:
            logger.debug("Abort session %s failed (best-effort)", session_id, exc_info=True)

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

    @staticmethod
    def _parse_model(model: str) -> dict[str, str]:
        """Convert 'providerID/modelID' string to the OpenCode model object.

        Args:
            model: Model string, e.g. "ollama-cloud/kimi-k2.5".

        Returns:
            Dict with 'providerID' and 'modelID' keys.
        """
        provider, sep, model_id = model.partition("/")
        if not sep:
            return {"providerID": "", "modelID": provider}
        return {"providerID": provider, "modelID": model_id}

    def _handle_sse_error(self, data: dict) -> None:  # type: ignore[type-arg]
        """Handle an SSE error event from the OpenCode stream.

        Args:
            data: The payload.properties dict from an "error" SSE event.

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
