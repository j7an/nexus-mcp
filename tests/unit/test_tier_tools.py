"""Unit tests for set_model_tiers tool and get_tiers_resource."""

from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.resources import get_tiers_resource
from nexus_mcp.server import set_model_tiers


class TestSetModelTiers:
    @patch("nexus_mcp.server.save_model_tiers")
    async def test_saves_tiers_and_returns_confirmation(self, mock_save, ctx):
        tiers = {"gemini-2.5-flash": "quick", "gemini-2.5-pro": "thorough"}
        result = await set_model_tiers(tiers=tiers, ctx=ctx)
        assert "2 model(s)" in result
        mock_save.assert_awaited_once_with(ctx, tiers)

    async def test_requires_context(self):
        with pytest.raises(ToolError, match="requires.*context"):
            await set_model_tiers(tiers={"a": "quick"}, ctx=None)

    @patch("nexus_mcp.server.save_model_tiers")
    async def test_empty_tiers_is_valid(self, mock_save, ctx):
        result = await set_model_tiers(tiers={}, ctx=ctx)
        assert "0 model(s)" in result
        mock_save.assert_awaited_once_with(ctx, {})


class TestGetTiersResource:
    @patch("nexus_mcp.resources.load_model_tiers")
    async def test_returns_saved_tiers(self, mock_load, ctx):
        import json

        mock_load.return_value = {"gemini-2.5-flash": "quick"}
        result = await get_tiers_resource(ctx=ctx)
        assert json.loads(result) == {"gemini-2.5-flash": "quick"}

    @patch("nexus_mcp.resources.load_model_tiers", return_value=None)
    async def test_returns_empty_dict_when_no_tiers(self, mock_load, ctx):
        import json

        result = await get_tiers_resource(ctx=ctx)
        assert json.loads(result) == {}

    async def test_returns_empty_dict_without_context(self):
        import json

        result = await get_tiers_resource(ctx=None)
        assert json.loads(result) == {}
