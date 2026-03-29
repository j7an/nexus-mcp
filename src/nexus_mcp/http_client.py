"""HTTP client for OpenCode server communication.

Single point of contact with opencode serve. Handles auth, health checks,
session management, SSE streaming, and error classification.

All httpx usage is contained here — the rest of the codebase never imports
httpx directly. This enables future swap to httpxyz if httpx remains stalled.
"""

import contextlib
import json
import logging

import httpx
from httpx_sse import aconnect_sse

from nexus_mcp.config_resolver import get_opencode_server_auth, get_opencode_server_url
from nexus_mcp.exceptions import RetryableError, SubprocessError

logger = logging.getLogger(__name__)

# Module-level singleton — initialized on first access via get_http_client().
_client: "OpenCodeHTTPClient | None" = None


def get_http_client() -> "OpenCodeHTTPClient":
    """Return the module-level OpenCodeHTTPClient singleton.

    Creates the client on first call. Subsequent calls return the same instance.
    """
    global _client  # noqa: PLW0603
    if _client is None:
        _client = OpenCodeHTTPClient()
    return _client


def reset_http_client() -> None:
    """Reset the singleton (for tests)."""
    global _client  # noqa: PLW0603
    _client = None


class OpenCodeHTTPClient:
    """HTTP client for opencode serve API.

    Wraps httpx.AsyncClient with Basic auth, health checking, and error
    classification. All HTTP communication with the OpenCode server flows
    through this class.
    """

    def __init__(self) -> None:
        url = get_opencode_server_url()
        username, password = get_opencode_server_auth()
        self._httpx = httpx.AsyncClient(
            base_url=url,
            auth=httpx.BasicAuth(username, password),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._session_cache: dict[str, str] = {}  # label → session_id

    async def health_check(self) -> bool:
        """Check if opencode serve is reachable.

        Returns:
            True if server responds 200 to GET /global/health, False otherwise.
        """
        try:
            response = await self._httpx.get("/global/health")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def classify_error(self, response: httpx.Response) -> None:
        """Raise appropriate exception based on HTTP status code.

        Args:
            response: HTTP response with non-2xx status.

        Raises:
            RetryableError: For 429 (rate limited) and 503 (unavailable).
            SubprocessError: For all other error status codes.
        """
        status = response.status_code
        retry_after: float | None = None

        if status == 429:
            raw = response.headers.get("Retry-After")
            if raw is not None:
                with contextlib.suppress(ValueError):
                    retry_after = float(raw)

        if status in (429, 503):
            raise RetryableError(
                f"OpenCode server returned {status}",
                retry_after=retry_after,
            )

        raise SubprocessError(f"OpenCode server returned {status}")

    async def get(self, path: str, **kwargs: object) -> object:
        """Send a GET request. Returns parsed JSON response.

        Raises SubprocessError or RetryableError on non-200 status.
        """
        response = await self._httpx.get(path, **kwargs)  # type: ignore[arg-type]
        if response.status_code != 200:
            self.classify_error(response)
        return response.json()

    async def put(self, path: str, **kwargs: object) -> object:
        """Send a PUT request. Returns parsed JSON response.

        Raises SubprocessError or RetryableError on non-200 status.
        """
        response = await self._httpx.put(path, **kwargs)  # type: ignore[arg-type]
        if response.status_code != 200:
            self.classify_error(response)
        return response.json()

    async def patch(self, path: str, **kwargs: object) -> object:
        """Send a PATCH request. Returns parsed JSON response.

        Raises SubprocessError or RetryableError on non-200 status.
        """
        response = await self._httpx.patch(path, **kwargs)  # type: ignore[arg-type]
        if response.status_code != 200:
            self.classify_error(response)
        return response.json()

    async def post(self, path: str, **kwargs: object) -> object:
        """Send a POST request. Returns parsed JSON response.

        Raises SubprocessError or RetryableError on non-200 status.
        """
        response = await self._httpx.post(path, **kwargs)  # type: ignore[arg-type]
        if response.status_code != 200:
            self.classify_error(response)
        return response.json()

    async def delete(self, path: str) -> None:
        """Send a DELETE request. Tolerates 404 (already gone)."""
        response = await self._httpx.delete(path)
        if response.status_code not in (200, 404):
            self.classify_error(response)

    async def create_session(self) -> str:
        """Create a new session on the OpenCode server.

        Returns:
            The session ID string.

        Raises:
            SubprocessError or RetryableError on HTTP errors.
        """
        response = await self._httpx.post("/session")
        if response.status_code != 200:
            self.classify_error(response)
        data: dict[str, str] = response.json()
        return data["id"]

    async def get_session(self, session_id: str) -> dict[str, object] | None:
        """Get session details. Returns None if session doesn't exist (404)."""
        response = await self._httpx.get(f"/session/{session_id}")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            self.classify_error(response)
        result: dict[str, object] = response.json()
        return result

    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        response = await self._httpx.delete(f"/session/{session_id}")
        if response.status_code not in (200, 404):
            self.classify_error(response)

    async def fork_session(self, session_id: str) -> str:
        """Fork a session at its current point.

        Returns:
            The new forked session ID.
        """
        response = await self._httpx.post(f"/session/{session_id}/fork")
        if response.status_code != 200:
            self.classify_error(response)
        data: dict[str, str] = response.json()
        return data["id"]

    async def resolve_session(self, label: str | None) -> str:
        """Resolve a session ID from a label, creating if needed.

        Strategy:
        - No label → create ephemeral session (not cached)
        - Label in cache → validate via GET, reuse if exists, evict + create if 404
        - Label not in cache → create new, cache it
        """
        if label is None:
            return await self.create_session()

        cached_id = self._session_cache.get(label)
        if cached_id is not None:
            existing = await self.get_session(cached_id)
            if existing is not None:
                return cached_id
            del self._session_cache[label]

        session_id = await self.create_session()
        self._session_cache[label] = session_id
        return session_id

    async def send_prompt(self, session_id: str, prompt_text: str) -> str:
        """Send a prompt and collect the response via SSE.

        Implements race condition mitigation from OpenCode PR #12965:
        1. Connect SSE (subscribe to events)
        2. Wait for server.connected
        3. POST prompt
        4. Collect part.updated text events
        5. Return when session.status = completed

        Args:
            session_id: The session to send the prompt to.
            prompt_text: The prompt text.

        Returns:
            Aggregated text from all part.updated events.
        """
        parts: list[str] = []
        prompt_sent = False

        async with aconnect_sse(self._httpx, "GET", "/event") as event_source:
            async for event in event_source.aiter_sse():
                if event.event == "server.connected" and not prompt_sent:
                    # Step 3: Send prompt AFTER subscribing (race condition fix)
                    response = await self._httpx.post(
                        f"/session/{session_id}/message",
                        json={"content": prompt_text},
                    )
                    if response.status_code != 200:
                        self.classify_error(response)
                    prompt_sent = True

                elif event.event == "part.updated":
                    try:
                        data = json.loads(event.data)
                        part = data.get("part", {})
                        if part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                parts.append(text)
                    except json.JSONDecodeError:
                        pass

                elif event.event == "session.status":
                    try:
                        data = json.loads(event.data)
                        if data.get("status") == "completed":
                            break
                    except json.JSONDecodeError:
                        pass

        return "".join(parts)

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._httpx.aclose()
