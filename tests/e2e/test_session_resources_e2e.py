# tests/e2e/test_session_resources_e2e.py
"""E2E tests for Phase 3 session resources via MCP protocol.

Tests resource discovery, URI template resolution, and content validation
for session-related resources when OpenCode server is healthy.
"""

import json

import httpx
import pytest
import respx
from fastmcp import Client
from mcp.shared.exceptions import McpError

from nexus_mcp.http_client import reset_http_client
from nexus_mcp.server import mcp

MINIMAL_OPENAPI_SPEC = {
    "openapi": "3.1.0",
    "info": {"title": "OpenCode", "version": "1.0.0"},
    "servers": [{"url": "http://test:4096"}],
    "paths": {},
}


@pytest.fixture
async def healthy_client(monkeypatch):
    """Client connected to a healthy OpenCode server."""
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
    reset_http_client()
    with respx.mock:
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=MINIMAL_OPENAPI_SPEC)
        )
        try:
            async with Client(mcp) as client:
                yield client
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()


@pytest.mark.e2e
class TestSessionResourceDiscovery:
    """Verify session resources appear in list_resources when server is healthy."""

    @respx.mock
    async def test_static_session_resources_listed(self, healthy_client):
        resources = await healthy_client.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "nexus://opencode/sessions" in uris
        assert "nexus://opencode/sessions/status" in uris
        assert "nexus://opencode/permissions" in uris
        assert "nexus://opencode/questions" in uris

    @respx.mock
    async def test_all_session_resources_have_json_mime_type(self, healthy_client):
        resources = await healthy_client.list_resources()
        session_resources = [
            r for r in resources if str(r.uri).startswith("nexus://opencode/session")
        ]
        for resource in session_resources:
            assert resource.mimeType == "application/json"

    @respx.mock
    async def test_all_session_resources_have_annotations(self, healthy_client):
        resources = await healthy_client.list_resources()
        session_resources = [
            r for r in resources if str(r.uri).startswith("nexus://opencode/session")
        ]
        for resource in session_resources:
            assert resource.annotations is not None
            assert resource.annotations.readOnlyHint is True
            assert resource.annotations.idempotentHint is True


@pytest.mark.e2e
class TestSessionTemplateDiscovery:
    """Verify URI template resources appear in list_resource_templates."""

    @respx.mock
    async def test_session_templates_listed(self, healthy_client):
        templates = await healthy_client.list_resource_templates()
        template_uris = {str(t.uriTemplate) for t in templates}
        assert "nexus://opencode/session/{session_id}/todo" in template_uris
        assert "nexus://opencode/session/{session_id}/messages" in template_uris
        assert "nexus://opencode/session/{session_id}/children" in template_uris
        assert "nexus://opencode/session/{session_id}/diff" in template_uris
        assert "nexus://opencode/session/{session_id}/message/{message_id}" in template_uris


@pytest.mark.e2e
class TestSessionResourceReading:
    """Verify session resources return correct data via read_resource."""

    @respx.mock
    async def test_read_sessions_returns_valid_json(self, healthy_client):
        respx.get("http://test:4096/session").mock(
            return_value=httpx.Response(200, json=[{"id": "ses_abc", "status": "idle"}])
        )
        contents = await healthy_client.read_resource("nexus://opencode/sessions")
        data = json.loads(contents[0].text)
        assert isinstance(data, list)
        assert data[0]["id"] == "ses_abc"

    @respx.mock
    async def test_read_permissions_returns_valid_json(self, healthy_client):
        respx.get("http://test:4096/permission").mock(return_value=httpx.Response(200, json=[]))
        contents = await healthy_client.read_resource("nexus://opencode/permissions")
        data = json.loads(contents[0].text)
        assert isinstance(data, list)

    @respx.mock
    async def test_read_session_todo_resolves_template(self, healthy_client):
        respx.get("http://test:4096/session/ses_abc/todo").mock(
            return_value=httpx.Response(200, json=[{"text": "Fix bug", "completed": False}])
        )
        contents = await healthy_client.read_resource("nexus://opencode/session/ses_abc/todo")
        data = json.loads(contents[0].text)
        assert data[0]["text"] == "Fix bug"

    @respx.mock
    async def test_read_session_message_resolves_double_template(self, healthy_client):
        respx.get("http://test:4096/session/ses_abc/message/msg_123").mock(
            return_value=httpx.Response(
                200, json={"id": "msg_123", "role": "user", "content": "hello"}
            )
        )
        contents = await healthy_client.read_resource(
            "nexus://opencode/session/ses_abc/message/msg_123"
        )
        data = json.loads(contents[0].text)
        assert data["content"] == "hello"

    @respx.mock
    async def test_invalid_session_id_raises_error(self, healthy_client):
        with pytest.raises(McpError):
            await healthy_client.read_resource("nexus://opencode/session/../../etc/passwd/todo")
