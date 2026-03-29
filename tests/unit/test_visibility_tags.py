"""Tests for FastMCP visibility tag assignments."""

import pytest


@pytest.fixture(autouse=True)
def _mock_cli(mock_cli_detection):
    yield mock_cli_detection


class TestToolTags:
    async def test_prompt_has_agent_execution_tag(self):
        from nexus_mcp.server import mcp

        tools = await mcp.list_tools()
        prompt_tool = next((t for t in tools if t.name == "prompt"), None)
        assert prompt_tool is not None
        tags = prompt_tool.tags
        assert "agent-execution" in tags

    async def test_batch_prompt_has_agent_execution_tag(self):
        from nexus_mcp.server import mcp

        tools = await mcp.list_tools()
        tool = next((t for t in tools if t.name == "batch_prompt"), None)
        assert tool is not None
        assert "agent-execution" in tool.tags

    async def test_set_preferences_has_configuration_tag(self):
        from nexus_mcp.server import mcp

        tools = await mcp.list_tools()
        tool = next((t for t in tools if t.name == "set_preferences"), None)
        assert tool is not None
        assert "configuration" in tool.tags

    async def test_get_preferences_has_configuration_tag(self):
        from nexus_mcp.server import mcp

        tools = await mcp.list_tools()
        tool = next((t for t in tools if t.name == "get_preferences"), None)
        assert tool is not None
        assert "configuration" in tool.tags
