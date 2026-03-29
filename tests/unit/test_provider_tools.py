"""Tests for hand-written provider configuration tools."""

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


class TestSetProviderAuth:
    @respx.mock
    async def test_sets_credentials(self):
        from nexus_mcp.server import opencode_set_provider_auth

        respx.put("http://test:4096/auth/anthropic").mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        result = await opencode_set_provider_auth(
            provider_id="anthropic", credentials={"apiKey": "sk-test"}
        )
        assert "anthropic" in result.lower()


class TestUpdateConfig:
    @respx.mock
    async def test_patches_config(self):
        from nexus_mcp.server import opencode_update_config

        respx.patch("http://test:4096/config").mock(
            return_value=httpx.Response(200, json={"model": "kimi-k2.5"})
        )
        result = await opencode_update_config(config={"model": "kimi-k2.5"})
        assert "kimi" in result
