"""Tests for OpenCode server MCP resources."""

import httpx
import pytest
import respx

from nexus_mcp.http_client import reset_http_client


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


class TestOpenCodeStatusResource:
    async def test_not_configured_returns_configured_false(self, monkeypatch):
        monkeypatch.delenv("NEXUS_OPENCODE_SERVER_PASSWORD", raising=False)
        import json

        from nexus_mcp.opencode_resources import get_opencode_status

        result = json.loads(await get_opencode_status())
        assert result["server"]["configured"] is False
        assert result["server"]["healthy"] is False
        assert result["server"]["url"] is None
        assert result["tool_groups"] == []
        assert result["compound_tools"] == []

    @respx.mock
    async def test_configured_healthy(self):
        respx.get("http://test:4096/global/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        import json

        from nexus_mcp.opencode_resources import get_opencode_status

        result = json.loads(await get_opencode_status())
        assert result["server"]["configured"] is True
        assert result["server"]["healthy"] is True
        assert result["server"]["url"] == "http://test:4096"

    @respx.mock
    async def test_configured_unhealthy(self):
        respx.get("http://test:4096/global/health").mock(return_value=httpx.Response(503))
        import json

        from nexus_mcp.opencode_resources import get_opencode_status

        result = json.loads(await get_opencode_status())
        assert result["server"]["configured"] is True
        assert result["server"]["healthy"] is False


class TestOpenCodeProvidersResource:
    @respx.mock
    async def test_returns_provider_list(self):
        respx.get("http://test:4096/provider").mock(
            return_value=httpx.Response(200, json=[{"id": "anthropic", "name": "Anthropic"}])
        )
        from nexus_mcp.opencode_resources import get_opencode_providers

        result = await get_opencode_providers()
        assert "anthropic" in result


class TestOpenCodeProvidersAuthResource:
    @respx.mock
    async def test_returns_auth_methods(self):
        respx.get("http://test:4096/provider/auth").mock(
            return_value=httpx.Response(200, json={"anthropic": {"type": "api_key"}})
        )
        from nexus_mcp.opencode_resources import get_opencode_providers_auth

        result = await get_opencode_providers_auth()
        assert "anthropic" in result


class TestOpenCodeConfigResource:
    @respx.mock
    async def test_returns_config(self):
        respx.get("http://test:4096/config").mock(
            return_value=httpx.Response(200, json={"model": "gpt-4"})
        )
        from nexus_mcp.opencode_resources import get_opencode_config

        result = await get_opencode_config()
        assert "model" in result
