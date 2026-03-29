"""HTTP client for OpenCode server communication.

Single point of contact with opencode serve. Handles auth, health checks,
session management, SSE streaming, and error classification.

All httpx usage is contained here — the rest of the codebase never imports
httpx directly. This enables future swap to httpxyz if httpx remains stalled.
"""

import contextlib
import logging

import httpx

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

    async def close(self) -> None:
        """Close the underlying httpx client."""
        await self._httpx.aclose()
