"""Tests for OpenCodeServerRunner."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from nexus_mcp.exceptions import RetryableError
from nexus_mcp.http_client import reset_http_client
from nexus_mcp.types import AgentResponse
from tests.fixtures import make_opencode_server_runner, make_prompt_request


@pytest.fixture(autouse=True)
def _reset_client():
    """Reset the HTTP client singleton between tests."""
    reset_http_client()
    yield
    reset_http_client()


@pytest.fixture
def runner(monkeypatch):
    return make_opencode_server_runner(monkeypatch)


class TestRunnerProperties:
    def test_agent_name(self, runner):
        assert runner.AGENT_NAME == "opencode_server"

    def test_supported_modes(self, runner):
        assert runner._SUPPORTED_MODES == ("default",)

    def test_build_command_raises(self, runner):
        request = make_prompt_request(cli="opencode_server")
        with pytest.raises(NotImplementedError):
            runner.build_command(request)

    def test_parse_output_raises(self, runner):
        with pytest.raises(NotImplementedError):
            runner.parse_output("", "")


class TestExecuteFlow:
    @respx.mock
    async def test_execute_full_flow(self, runner):
        """Health check → session → SSE subscribe → prompt → collect → response."""
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_test"})
        )
        respx.post("http://test:4096/session/ses_test/message").mock(
            return_value=httpx.Response(200)
        )

        mock_events = [
            MagicMock(event="server.connected", data="{}"),
            MagicMock(event="part.updated", data='{"part": {"type": "text", "text": "pong"}}'),
            MagicMock(event="session.status", data='{"status": "completed"}'),
        ]

        async def mock_aiter():
            for e in mock_events:
                yield e

        mock_source = AsyncMock()
        mock_source.aiter_sse = mock_aiter
        mock_source.__aenter__ = AsyncMock(return_value=mock_source)
        mock_source.__aexit__ = AsyncMock(return_value=False)

        request = make_prompt_request(cli="opencode_server", prompt="ping")
        emit = AsyncMock()
        progress = AsyncMock()

        with patch("nexus_mcp.http_client.aconnect_sse", return_value=mock_source):
            response = await runner._execute(request, emit, progress)

        assert isinstance(response, AgentResponse)
        assert response.cli == "opencode_server"
        assert response.output == "pong"

    @respx.mock
    async def test_execute_health_check_fails_raises_retryable(self, runner):
        """Health check failure → RetryableError (server not reachable)."""
        respx.get("http://test:4096/global/health").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        request = make_prompt_request(cli="opencode_server")
        emit = AsyncMock()
        progress = AsyncMock()

        with pytest.raises(RetryableError, match="not reachable"):
            await runner._execute(request, emit, progress)
