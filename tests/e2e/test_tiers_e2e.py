"""E2E tests for model tier tools and nexus://tiers resource via MCP protocol."""

import json

import pytest
from fastmcp import Client

from nexus_mcp.server import mcp


@pytest.mark.e2e
class TestTierToolDiscovery:
    async def test_set_model_tiers_in_tool_list(self, mcp_client):
        tools = await mcp_client.list_tools()
        names = {t.name for t in tools}
        assert "set_model_tiers" in names

    async def test_get_model_tiers_not_in_tool_list(self, mcp_client):
        """get_model_tiers was migrated to nexus://tiers resource."""
        tools = await mcp_client.list_tools()
        names = {t.name for t in tools}
        assert "get_model_tiers" not in names


@pytest.mark.e2e
class TestTierToolAnnotations:
    async def test_set_model_tiers_annotations(self, mcp_client):
        tools = await mcp_client.list_tools()
        tool = next(t for t in tools if t.name == "set_model_tiers")
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.title == "Set Model Tiers"


@pytest.mark.e2e
class TestTierToolRoundTrip:
    async def test_get_returns_empty_when_no_tiers_set(self, mcp_client):
        contents = await mcp_client.read_resource("nexus://tiers")
        assert json.loads(contents[0].text) == {}

    async def test_set_then_get_round_trips(self, mcp_client):
        tiers = {"gemini-2.5-flash": "quick", "gemini-2.5-pro": "thorough"}
        await mcp_client.call_tool("set_model_tiers", {"tiers": tiers})
        contents = await mcp_client.read_resource("nexus://tiers")
        assert json.loads(contents[0].text) == tiers

    async def test_set_overwrites_previous(self, mcp_client):
        await mcp_client.call_tool("set_model_tiers", {"tiers": {"model-a": "quick"}})
        await mcp_client.call_tool("set_model_tiers", {"tiers": {"model-b": "thorough"}})
        contents = await mcp_client.read_resource("nexus://tiers")
        assert json.loads(contents[0].text) == {"model-b": "thorough"}


@pytest.mark.e2e
class TestTierPersistenceAcrossSessions:
    async def test_tiers_persist_across_sessions(self):
        tiers = {"gemini-2.5-flash": "quick", "gpt-5.2": "standard"}
        try:
            async with Client(mcp) as client1:
                await client1.call_tool("set_model_tiers", {"tiers": tiers})
        finally:
            mcp._lifespan_result_set = False

        try:
            async with Client(mcp) as client2:
                contents = await client2.read_resource("nexus://tiers")
                assert json.loads(contents[0].text) == tiers
        finally:
            mcp._lifespan_result_set = False
