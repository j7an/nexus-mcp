"""Tests for OpenCodeHTTPClient — auth, health check, error classification."""

import httpx
import pytest
import respx

from nexus_mcp.exceptions import RetryableError, SubprocessError
from nexus_mcp.http_client import OpenCodeHTTPClient, reset_http_client


@pytest.fixture(autouse=True)
def _reset():
    reset_http_client()
    yield
    reset_http_client()


@pytest.fixture
def client(monkeypatch):
    """Create an OpenCodeHTTPClient with test credentials."""
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_USERNAME", "testuser")
    return OpenCodeHTTPClient()


class TestAuth:
    def test_client_uses_basic_auth(self, client):
        """Client is configured with HTTP Basic auth from env vars."""
        assert client._httpx.auth is not None

    def test_base_url(self, client):
        assert str(client._httpx.base_url) == "http://test:4096"


class TestHealthCheck:
    @respx.mock
    async def test_health_check_success(self, client):
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        result = await client.health_check()
        assert result is True

    @respx.mock
    async def test_health_check_failure_returns_false(self, client):
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(503, json={"error": "starting"})
        )
        result = await client.health_check()
        assert result is False

    @respx.mock
    async def test_health_check_connect_error_returns_false(self, client):
        respx.get("http://test:4096/global/health").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        result = await client.health_check()
        assert result is False


class TestErrorClassification:
    def test_429_raises_retryable(self, client):
        response = httpx.Response(429, headers={"Retry-After": "5"})
        with pytest.raises(RetryableError) as exc_info:
            client.classify_error(response)
        assert exc_info.value.retry_after == 5.0

    def test_503_raises_retryable(self, client):
        response = httpx.Response(503)
        with pytest.raises(RetryableError):
            client.classify_error(response)

    def test_401_raises_subprocess_error(self, client):
        response = httpx.Response(401)
        with pytest.raises(SubprocessError, match="401"):
            client.classify_error(response)

    def test_404_raises_subprocess_error(self, client):
        response = httpx.Response(404)
        with pytest.raises(SubprocessError, match="404"):
            client.classify_error(response)

    def test_500_raises_subprocess_error(self, client):
        response = httpx.Response(500)
        with pytest.raises(SubprocessError, match="500"):
            client.classify_error(response)


class TestSessionManagement:
    @respx.mock
    async def test_create_session(self, client):
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_123"})
        )
        session_id = await client.create_session()
        assert session_id == "ses_123"

    @respx.mock
    async def test_get_session_exists(self, client):
        respx.get("http://test:4096/session/ses_123").mock(
            return_value=httpx.Response(200, json={"id": "ses_123", "status": "active"})
        )
        result = await client.get_session("ses_123")
        assert result["id"] == "ses_123"

    @respx.mock
    async def test_get_session_not_found(self, client):
        respx.get("http://test:4096/session/ses_gone").mock(return_value=httpx.Response(404))
        result = await client.get_session("ses_gone")
        assert result is None

    @respx.mock
    async def test_delete_session(self, client):
        respx.delete("http://test:4096/session/ses_123").mock(return_value=httpx.Response(200))
        await client.delete_session("ses_123")

    @respx.mock
    async def test_fork_session(self, client):
        respx.post("http://test:4096/session/ses_123/fork").mock(
            return_value=httpx.Response(200, json={"id": "ses_fork_456"})
        )
        forked_id = await client.fork_session("ses_123")
        assert forked_id == "ses_fork_456"

    @respx.mock
    async def test_resolve_session_creates_on_cache_miss(self, client):
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_new"})
        )
        session_id = await client.resolve_session("my-label")
        assert session_id == "ses_new"

    @respx.mock
    async def test_resolve_session_reuses_cached(self, client):
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_cached"})
        )
        respx.get("http://test:4096/session/ses_cached").mock(
            return_value=httpx.Response(200, json={"id": "ses_cached"})
        )
        first = await client.resolve_session("reuse-label")
        assert first == "ses_cached"
        second = await client.resolve_session("reuse-label")
        assert second == "ses_cached"

    @respx.mock
    async def test_resolve_session_evicts_on_404(self, client):
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_first"})
        )
        await client.resolve_session("evict-label")

        respx.get("http://test:4096/session/ses_first").mock(return_value=httpx.Response(404))
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_second"})
        )
        second = await client.resolve_session("evict-label")
        assert second == "ses_second"

    @respx.mock
    async def test_resolve_session_no_label_creates_ephemeral(self, client):
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_ephemeral"})
        )
        session_id = await client.resolve_session(None)
        assert session_id == "ses_ephemeral"
