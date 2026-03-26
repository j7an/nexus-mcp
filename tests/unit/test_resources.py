# tests/unit/test_resources.py
"""Unit tests for MCP resource functions.

Tests call resource functions directly (no MCP protocol).
Mock boundary: detect_cli, get_cli_version for runner detection.
"""

import json
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ResourceError

from nexus_mcp.resources import get_all_runners, get_config, get_preferences_resource, get_runner

RUNNER_INFO_KEYS = {
    "name",
    "installed",
    "path",
    "version",
    "models",
    "default_model",
    "supported_modes",
    "default_timeout",
}


class TestGetAllRunners:
    """Tests for the nexus://runners resource."""

    async def test_returns_all_registered_runners(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        names = {r["name"] for r in result["runners"]}
        assert names == {"claude", "codex", "gemini", "opencode"}

    async def test_json_structure_has_required_keys(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        assert "runners" in result
        for runner in result["runners"]:
            assert set(runner.keys()) == RUNNER_INFO_KEYS

    async def test_installed_runner_has_path(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        gemini = next(r for r in result["runners"] if r["name"] == "gemini")
        assert gemini["installed"] is True
        assert gemini["path"] is not None

    async def test_supported_modes_match_runner_class(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        gemini = next(r for r in result["runners"] if r["name"] == "gemini")
        assert gemini["supported_modes"] == ["default", "yolo"]
        opencode = next(r for r in result["runners"] if r["name"] == "opencode")
        assert opencode["supported_modes"] == ["default"]


class TestGetRunner:
    """Tests for the nexus://runners/{cli} template resource."""

    async def test_returns_correct_runner(self, mock_cli_detection):
        result = json.loads(await get_runner("gemini"))
        assert result["name"] == "gemini"
        assert set(result.keys()) == RUNNER_INFO_KEYS

    async def test_not_found_raises_resource_error(self, mock_cli_detection):
        with pytest.raises(ResourceError, match="unknown_cli"):
            await get_runner("unknown_cli")

    async def test_uninstalled_cli_returns_false_installed(self, mock_cli_detection):
        """When detect_cli returns found=False, installed/path/version are None/False."""
        from unittest.mock import patch

        from nexus_mcp.cli_detector import CLIInfo

        with patch("nexus_mcp.resources.detect_cli", return_value=CLIInfo(found=False)):
            result = json.loads(await get_runner("gemini"))
        assert result["installed"] is False
        assert result["path"] is None
        assert result["version"] is None


CONFIG_KEYS = {
    "timeout",
    "output_limit",
    "max_retries",
    "retry_base_delay",
    "retry_max_delay",
    "tool_timeout",
    "cli_detection_timeout",
}


class TestGetConfig:
    """Tests for the nexus://config resource."""

    async def test_returns_resolved_defaults(self):
        result = json.loads(await get_config())
        assert set(result.keys()) == CONFIG_KEYS

    async def test_all_fields_non_null(self):
        result = json.loads(await get_config())
        for key, value in result.items():
            assert value is not None, f"Config field {key!r} is None"

    async def test_values_match_hardcoded_defaults(self):
        """Without env var overrides, values match HARDCODED_DEFAULTS."""
        from nexus_mcp.config_resolver import HARDCODED_DEFAULTS

        result = json.loads(await get_config())
        assert result["timeout"] == HARDCODED_DEFAULTS.timeout
        assert result["max_retries"] == HARDCODED_DEFAULTS.max_retries
        assert result["retry_base_delay"] == HARDCODED_DEFAULTS.retry_base_delay


class TestGetPreferencesResource:
    """Tests for the nexus://preferences resource."""

    @patch("nexus_mcp.preferences.load_preferences")
    async def test_session_preferences_returned(self, mock_load, ctx):
        """When session has preferences, returns them with source='session'."""
        mock_load.return_value = {"execution_mode": "yolo", "model": "gemini-2.5-pro"}
        result = json.loads(await get_preferences_resource(ctx=ctx))
        assert result["source"] == "session"
        assert result["preferences"]["execution_mode"] == "yolo"
        assert result["preferences"]["model"] == "gemini-2.5-pro"

    @patch("nexus_mcp.preferences.load_preferences", return_value=None)
    async def test_empty_session_returns_session_source(self, _mock_load, ctx):
        """When session exists but no prefs set, returns empty prefs with source='session'."""
        result = json.loads(await get_preferences_resource(ctx=ctx))
        assert result["source"] == "session"
        assert result["preferences"]["execution_mode"] is None

    async def test_fallback_to_defaults_when_no_ctx(self):
        """When ctx is None, falls back to config defaults with source='defaults'."""
        result = json.loads(await get_preferences_resource(ctx=None))
        assert result["source"] == "defaults"
        # Config defaults have non-None values for operational fields
        assert result["preferences"]["timeout"] is not None
        assert result["preferences"]["max_retries"] is not None

    @patch("nexus_mcp.preferences.load_preferences", side_effect=RuntimeError("store broken"))
    async def test_fallback_on_session_error(self, _mock_load, ctx):
        """When _get_session_preferences raises, falls back to config defaults."""
        result = json.loads(await get_preferences_resource(ctx=ctx))
        assert result["source"] == "defaults"
