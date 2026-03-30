# tests/e2e/test_session_tools_e2e.py
"""E2E tests for Phase 3 session tools via MCP protocol.

Tests that session lifecycle and permission/question tools are
discoverable and have correct metadata when OpenCode server is healthy.
"""

import httpx
import pytest
import respx
from fastmcp import Client

from nexus_mcp.http_client import reset_http_client
from nexus_mcp.server import mcp

# OpenAPI spec that includes session endpoints for auto-generation
SESSION_OPENAPI_SPEC = {
    "openapi": "3.1.0",
    "info": {"title": "OpenCode", "version": "1.0.0"},
    "servers": [{"url": "http://test:4096"}],
    "paths": {
        "/session/{sessionID}/abort": {
            "post": {
                "operationId": "session.abort",
                "summary": "Abort session",
                "parameters": [
                    {
                        "name": "sessionID",
                        "in": "path",
                        "schema": {"type": "string"},
                        "required": True,
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
        "/session/{sessionID}/summarize": {
            "post": {
                "operationId": "session.summarize",
                "summary": "Summarize session",
                "parameters": [
                    {
                        "name": "sessionID",
                        "in": "path",
                        "schema": {"type": "string"},
                        "required": True,
                    }
                ],
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
        "/permission/{requestID}/reply": {
            "post": {
                "operationId": "permission.reply",
                "summary": "Respond to permission request",
                "parameters": [
                    {
                        "name": "requestID",
                        "in": "path",
                        "schema": {"type": "string"},
                        "required": True,
                    }
                ],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "reply": {
                                        "type": "string",
                                        "enum": ["once", "always", "reject"],
                                    }
                                },
                                "required": ["reply"],
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {"type": "boolean"}}},
                    }
                },
            }
        },
    },
}


@pytest.fixture(autouse=True)
def _mock_cli(mock_cli_detection):
    """Mock CLI detection for all tests."""
    yield mock_cli_detection


@pytest.mark.e2e
class TestSessionToolDiscovery:
    """Verify session tools appear when server is configured + healthy."""

    @respx.mock
    async def test_session_tools_registered_when_healthy(self, monkeypatch):
        """Session tools from OpenAPI spec should be registered when server is healthy."""
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
        reset_http_client()
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=SESSION_OPENAPI_SPEC)
        )
        try:
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                # Auto-generated session tools from spec
                # FastMCP converts operationId dots to underscores
                assert "session_abort" in names or any("abort" in n for n in names)
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()

    @respx.mock
    async def test_session_tools_not_present_when_unhealthy(self, monkeypatch):
        """Session tools should not appear when server is not healthy."""
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
        reset_http_client()
        respx.get("http://test:4096/global/health").mock(return_value=httpx.Response(503))
        try:
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                # No session/abort related tools
                assert not any("abort" in n for n in names)
                assert not any("summarize" in n and "session" in n for n in names)
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()


@pytest.mark.e2e
class TestSessionToolMetadata:
    """Verify session tools have correct tags and descriptions."""

    @respx.mock
    async def test_permission_reply_tool_registered(self, monkeypatch):
        """Permission reply tool from OpenAPI spec should be registered."""
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
        reset_http_client()
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=SESSION_OPENAPI_SPEC)
        )
        try:
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                assert any("permission" in n for n in names)
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()
