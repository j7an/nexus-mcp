# tests/e2e/test_conditional_registration.py
"""E2E tests for conditional OpenCode tool registration.

Tests the three lifespan states: not configured, configured+unhealthy,
configured+healthy. Each test creates a fresh Client(mcp) with appropriate
env vars and HTTP mocks.
"""

import json

import httpx
import pytest
import respx
from fastmcp import Client

from nexus_mcp.http_client import reset_http_client
from nexus_mcp.server import mcp

# Minimal valid OpenAPI spec for testing
MINIMAL_OPENAPI_SPEC = {
    "openapi": "3.1.0",
    "info": {"title": "OpenCode", "version": "1.0.0"},
    "servers": [{"url": "http://test:4096"}],
    "paths": {
        "/find": {
            "get": {
                "operationId": "find",
                "summary": "Search project files",
                "parameters": [{"name": "query", "in": "query", "schema": {"type": "string"}}],
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
    },
}


@pytest.fixture(autouse=True)
def _mock_cli(mock_cli_detection):
    yield mock_cli_detection


@pytest.mark.e2e
class TestNotConfiguredState:
    """Verify behavior when NEXUS_OPENCODE_SERVER_PASSWORD is NOT set."""

    async def test_only_core_tools_registered(self, monkeypatch):
        monkeypatch.delenv("NEXUS_OPENCODE_SERVER_PASSWORD", raising=False)
        reset_http_client()
        try:
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                # Only core tools — no OpenCode tools
                assert "prompt" in names
                assert "batch_prompt" in names
                assert "opencode_set_provider_auth" not in names
                assert "opencode_update_config" not in names
                assert "opencode_investigate" not in names
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()

    async def test_opencode_status_shows_not_configured(self, monkeypatch):
        monkeypatch.delenv("NEXUS_OPENCODE_SERVER_PASSWORD", raising=False)
        reset_http_client()
        try:
            async with Client(mcp) as client:
                contents = await client.read_resource("nexus://opencode")
                data = json.loads(contents[0].text)
                assert data["server"]["configured"] is False
                assert data["tool_groups"] == []
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()


@pytest.mark.e2e
class TestConfiguredUnhealthyState:
    """Verify behavior when server is configured but not reachable."""

    @respx.mock
    async def test_mutation_tools_registered_but_no_workspace(self, monkeypatch):
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
        reset_http_client()
        # Health check fails
        respx.get("http://test:4096/global/health").mock(return_value=httpx.Response(503))
        try:
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                # Core tools present
                assert "prompt" in names
                # Mutation tools present (server may recover)
                assert "opencode_set_provider_auth" in names
                assert "opencode_update_config" in names
                # Workspace/compound tools NOT present
                assert "opencode_investigate" not in names
                assert "opencode_session_review" not in names
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()

    @respx.mock
    async def test_opencode_status_shows_unhealthy(self, monkeypatch):
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
        reset_http_client()
        respx.get("http://test:4096/global/health").mock(return_value=httpx.Response(503))
        try:
            async with Client(mcp) as client:
                contents = await client.read_resource("nexus://opencode")
                data = json.loads(contents[0].text)
                assert data["server"]["configured"] is True
                assert data["server"]["healthy"] is False
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()


@pytest.mark.e2e
class TestConfiguredHealthyState:
    """Verify behavior when server is configured and reachable."""

    @respx.mock
    async def test_all_tools_registered(self, monkeypatch):
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
        reset_http_client()
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=MINIMAL_OPENAPI_SPEC)
        )
        try:
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = {t.name for t in tools}
                # Core tools
                assert "prompt" in names
                # Mutation tools
                assert "opencode_set_provider_auth" in names
                assert "opencode_update_config" in names
                # Compound tools
                assert "opencode_investigate" in names
                assert "opencode_session_review" in names
                # Auto-generated from OpenAPI spec
                assert "find" in names  # from MINIMAL_OPENAPI_SPEC
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()

    @respx.mock
    async def test_data_resources_registered(self, monkeypatch):
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
        reset_http_client()
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=MINIMAL_OPENAPI_SPEC)
        )
        try:
            async with Client(mcp) as client:
                resources = await client.list_resources()
                uris = {str(r.uri) for r in resources}
                assert "nexus://opencode" in uris
                assert "nexus://opencode/providers" in uris
                assert "nexus://opencode/providers/auth" in uris
                assert "nexus://opencode/config" in uris
        finally:
            mcp._lifespan_result_set = False
            reset_http_client()
