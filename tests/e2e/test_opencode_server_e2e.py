"""E2E tests: OpenCodeServerRunner through MCP protocol."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from nexus_mcp.http_client import reset_http_client
from nexus_mcp.server import batch_prompt
from nexus_mcp.types import AgentTask


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
    reset_http_client()
    yield
    reset_http_client()


def _mock_sse_source(text_response: str) -> AsyncMock:
    """Build a mock SSE context manager that yields canned events."""
    mock_events = [
        MagicMock(event="server.connected", data="{}"),
        MagicMock(
            event="part.updated",
            data=f'{{"part": {{"type": "text", "text": "{text_response}"}}}}',
        ),
        MagicMock(event="session.status", data='{"status": "completed"}'),
    ]

    async def mock_aiter():
        for e in mock_events:
            yield e

    mock_source = AsyncMock()
    mock_source.aiter_sse = mock_aiter
    mock_source.__aenter__ = AsyncMock(return_value=mock_source)
    mock_source.__aexit__ = AsyncMock(return_value=False)
    return mock_source


@pytest.mark.e2e
class TestOpenCodeServerE2E:
    @respx.mock
    async def test_batch_prompt_direct_opencode_server(self):
        """Direct batch_prompt call → OpenCodeServerRunner → mocked HTTP."""
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_e2e"})
        )
        respx.post("http://test:4096/session/ses_e2e/message").mock(
            return_value=httpx.Response(200)
        )

        mock_source = _mock_sse_source("e2e response")

        with patch("nexus_mcp.http_client.aconnect_sse", return_value=mock_source):
            result = await batch_prompt(
                tasks=[AgentTask(cli="opencode_server", prompt="test prompt")],
                elicit=False,
                ctx=None,
            )

        assert result.total == 1
        assert result.succeeded == 1
        assert result.results[0].success is True
        assert "e2e response" in result.results[0].output

    @respx.mock
    async def test_batch_prompt_includes_runner_header(self):
        """Response includes metadata header identifying the runner."""
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_hdr"})
        )
        respx.post("http://test:4096/session/ses_hdr/message").mock(
            return_value=httpx.Response(200)
        )

        mock_source = _mock_sse_source("header test")

        with patch("nexus_mcp.http_client.aconnect_sse", return_value=mock_source):
            result = await batch_prompt(
                tasks=[AgentTask(cli="opencode_server", prompt="check header")],
                elicit=False,
                ctx=None,
            )

        output = result.results[0].output
        assert "opencode_server" in output
        assert "header test" in output

    @respx.mock
    async def test_batch_prompt_health_check_failure_returns_error(self):
        """Health check failure → task result has error (RetryableError exhausted)."""
        respx.get("http://test:4096/global/health").mock(
            side_effect=httpx.ConnectError("connection refused")
        )

        result = await batch_prompt(
            tasks=[AgentTask(cli="opencode_server", prompt="test")],
            elicit=False,
            ctx=None,
        )

        assert result.total == 1
        assert result.failed == 1
        assert result.results[0].success is False
        assert result.results[0].error is not None

    @respx.mock
    async def test_prompt_via_mcp_protocol(self):
        """Full E2E: client → MCP prompt tool → OpenCodeServerRunner → mocked HTTP."""
        from fastmcp import Client

        from nexus_mcp.server import mcp

        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.post("http://test:4096/session").mock(
            return_value=httpx.Response(200, json={"id": "ses_e2e"})
        )
        respx.post("http://test:4096/session/ses_e2e/message").mock(
            return_value=httpx.Response(200)
        )

        mock_source = _mock_sse_source("e2e response")

        try:
            with patch("nexus_mcp.http_client.aconnect_sse", return_value=mock_source):
                async with Client(mcp) as client:
                    result = await client.call_tool(
                        "prompt",
                        {"cli": "opencode_server", "prompt": "test prompt", "elicit": False},
                    )
            assert result.is_error is False
            assert "e2e response" in result.data
        finally:
            mcp._lifespan_result_set = False
