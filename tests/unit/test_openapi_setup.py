"""Tests for openapi_setup — fetching OpenAPI spec and creating OpenAPIProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from fastmcp import FastMCP
from fastmcp.server.providers.openapi import MCPType

from nexus_mcp.http_client import OpenCodeHTTPClient, reset_http_client
from nexus_mcp.openapi_setup import (
    ROUTE_MAPS,
    fetch_openapi_spec,
    setup_opencode_tools,
)

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
        "/global/health": {
            "get": {
                "operationId": "global_health",
                "summary": "Health check",
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
        "/experimental/test": {
            "get": {
                "operationId": "experimental_test",
                "summary": "Experimental endpoint",
                "responses": {
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
        "/provider": {
            "get": {
                "operationId": "list_providers",
                "summary": "List providers",
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


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch):
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
    reset_http_client()
    yield
    reset_http_client()


@pytest.fixture
def client(monkeypatch):
    """Create an OpenCodeHTTPClient with test credentials."""
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_USERNAME", "testuser")
    return OpenCodeHTTPClient()


class TestFetchOpenAPISpec:
    @respx.mock
    async def test_success_returns_spec_dict(self, client):
        """fetch_openapi_spec returns the spec dict on success."""
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=MINIMAL_OPENAPI_SPEC)
        )
        result = await fetch_openapi_spec(client)
        assert result is not None
        assert result["openapi"] == "3.1.0"
        assert "paths" in result

    @respx.mock
    async def test_http_error_returns_none(self, client):
        """fetch_openapi_spec returns None on HTTP error."""
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(500, json={"error": "internal"})
        )
        result = await fetch_openapi_spec(client)
        assert result is None

    @respx.mock
    async def test_connect_error_returns_none(self, client):
        """fetch_openapi_spec returns None on connection failure."""
        respx.get("http://test:4096/doc").mock(side_effect=httpx.ConnectError("connection refused"))
        result = await fetch_openapi_spec(client)
        assert result is None

    @respx.mock
    async def test_missing_openapi_key_returns_none(self, client):
        """fetch_openapi_spec returns None if response lacks 'openapi' key."""
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json={"info": "not a spec"})
        )
        result = await fetch_openapi_spec(client)
        assert result is None

    @respx.mock
    async def test_uses_singleton_client_when_none_given(self):
        """fetch_openapi_spec uses get_http_client() when client is None."""
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=MINIMAL_OPENAPI_SPEC)
        )
        result = await fetch_openapi_spec(None)
        assert result is not None
        assert result["openapi"] == "3.1.0"


class TestRouteMapFiltering:
    def test_workspace_endpoints_included(self):
        """ROUTE_MAPS includes workspace-tagged entries for key endpoints."""
        workspace_maps = [rm for rm in ROUTE_MAPS if "workspace" in rm.mcp_tags]
        workspace_patterns = {str(rm.pattern) for rm in workspace_maps}
        assert r"/find$" in workspace_patterns
        assert r"/file$" in workspace_patterns
        assert r"/vcs$" in workspace_patterns
        assert r"/project$" in workspace_patterns

    def test_catch_all_exclude_exists_and_is_last(self):
        """ROUTE_MAPS ends with a catch-all EXCLUDE rule."""
        last = ROUTE_MAPS[-1]
        assert last.mcp_type == MCPType.EXCLUDE
        assert str(last.pattern) == r".*"

    def test_monitoring_endpoints_count(self):
        """ROUTE_MAPS has exactly 4 monitoring-tagged entries."""
        monitoring_maps = [rm for rm in ROUTE_MAPS if "monitoring" in rm.mcp_tags]
        assert len(monitoring_maps) == 4

    def test_monitoring_patterns(self):
        """Monitoring maps include health, lsp, mcp, formatter."""
        monitoring_maps = [rm for rm in ROUTE_MAPS if "monitoring" in rm.mcp_tags]
        patterns = {str(rm.pattern) for rm in monitoring_maps}
        assert r"/global/health$" in patterns
        assert r"/lsp$" in patterns
        assert r"/mcp$" in patterns
        assert r"/formatter$" in patterns

    def test_all_non_exclude_maps_are_tools(self):
        """All non-exclude route maps produce TOOL type."""
        non_exclude = [rm for rm in ROUTE_MAPS if rm.mcp_type != MCPType.EXCLUDE]
        for rm in non_exclude:
            assert rm.mcp_type == MCPType.TOOL

    def test_session_endpoints_included(self):
        """ROUTE_MAPS includes session-tagged entries for lifecycle endpoints."""
        session_maps = [rm for rm in ROUTE_MAPS if "session" in rm.mcp_tags]
        patterns = {str(rm.pattern) for rm in session_maps}
        assert r"/session/[^/]+/abort$" in patterns
        assert r"/session/[^/]+/summarize$" in patterns
        assert r"/session/[^/]+/share$" in patterns
        assert r"/session/[^/]+/revert$" in patterns
        assert r"/session/[^/]+/unrevert$" in patterns
        assert r"/session/[^/]+/command$" in patterns
        assert r"/session/[^/]+/init$" in patterns

    def test_session_update_endpoint_included(self):
        """ROUTE_MAPS includes PATCH /session/{id} for session update."""
        session_maps = [rm for rm in ROUTE_MAPS if "session" in rm.mcp_tags]
        patch_maps = [rm for rm in session_maps if "PATCH" in rm.methods]
        patterns = {str(rm.pattern) for rm in patch_maps}
        assert r"/session/[^/]+$" in patterns

    def test_permission_question_endpoints_included(self):
        """ROUTE_MAPS includes permission and question reply/reject endpoints."""
        session_maps = [rm for rm in ROUTE_MAPS if "session" in rm.mcp_tags]
        patterns = {str(rm.pattern) for rm in session_maps}
        assert r"/permission/[^/]+/reply$" in patterns
        assert r"/question/[^/]+/reply$" in patterns
        assert r"/question/[^/]+/reject$" in patterns

    def test_message_part_delete_endpoint_included(self):
        """ROUTE_MAPS includes DELETE for message part deletion."""
        session_maps = [rm for rm in ROUTE_MAPS if "session" in rm.mcp_tags]
        delete_maps = [rm for rm in session_maps if "DELETE" in rm.methods]
        patterns = {str(rm.pattern) for rm in delete_maps}
        assert r"/session/[^/]+/message/[^/]+/part/[^/]+$" in patterns

    def test_share_route_has_post_and_delete_methods(self):
        """Share route map includes both POST (share) and DELETE (unshare) methods."""
        session_maps = [rm for rm in ROUTE_MAPS if "session" in rm.mcp_tags]
        share_maps = [rm for rm in session_maps if "/share$" in str(rm.pattern)]
        assert len(share_maps) == 1
        assert "POST" in share_maps[0].methods
        assert "DELETE" in share_maps[0].methods

    def test_deprecated_session_permissions_excluded(self):
        """Deprecated POST /session/{id}/permissions/{id} is caught by EXCLUDE."""
        session_maps = [rm for rm in ROUTE_MAPS if "session" in rm.mcp_tags]
        patterns = {str(rm.pattern) for rm in session_maps}
        # No pattern should match the deprecated session-scoped permissions endpoint
        assert not any("permissions" in p for p in patterns)

    def test_session_endpoint_count(self):
        """ROUTE_MAPS has exactly 12 session-tagged entries (13 tools from 12 maps)."""
        session_maps = [rm for rm in ROUTE_MAPS if "session" in rm.mcp_tags]
        assert len(session_maps) == 12


class TestSetupOpenCodeTools:
    @respx.mock
    async def test_creates_provider_and_returns_true(self, client):
        """setup_opencode_tools returns True when spec is valid."""
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=MINIMAL_OPENAPI_SPEC)
        )
        mcp = MagicMock(spec=FastMCP)
        result = await setup_opencode_tools(mcp, client)
        assert result is True
        mcp.add_provider.assert_called_once()

    @respx.mock
    async def test_returns_false_on_fetch_failure(self, client):
        """setup_opencode_tools returns False when /doc is unreachable."""
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(500, json={"error": "down"})
        )
        mcp = MagicMock(spec=FastMCP)
        result = await setup_opencode_tools(mcp, client)
        assert result is False
        mcp.add_provider.assert_not_called()

    async def test_returns_false_on_invalid_spec(self, client):
        """setup_opencode_tools returns False when spec is malformed."""
        invalid_spec = {"openapi": "3.1.0"}  # missing required fields

        with (
            patch(
                "nexus_mcp.openapi_setup.fetch_openapi_spec",
                new=AsyncMock(return_value=invalid_spec),
            ),
            patch(
                "nexus_mcp.openapi_setup.OpenAPIProvider",
                side_effect=ValueError("invalid spec"),
            ),
        ):
            mcp = MagicMock(spec=FastMCP)
            result = await setup_opencode_tools(mcp, client)
            assert result is False
            mcp.add_provider.assert_not_called()

    @respx.mock
    async def test_provider_receives_httpx_client(self, client):
        """setup_opencode_tools passes the shared httpx client to OpenAPIProvider."""
        respx.get("http://test:4096/doc").mock(
            return_value=httpx.Response(200, json=MINIMAL_OPENAPI_SPEC)
        )
        mcp = MagicMock(spec=FastMCP)

        with patch("nexus_mcp.openapi_setup.OpenAPIProvider") as mock_provider_cls:
            mock_provider_cls.return_value = MagicMock()
            result = await setup_opencode_tools(mcp, client)

        assert result is True
        call_kwargs = mock_provider_cls.call_args
        assert call_kwargs.kwargs["client"] is client._httpx
