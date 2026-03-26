# tests/unit/test_resources.py
"""Unit tests for MCP resource functions.

Tests call resource functions directly (no MCP protocol).
Mock boundary: detect_cli, get_cli_version for runner detection.
"""

import json

import pytest
from fastmcp.exceptions import ResourceError

from nexus_mcp.resources import get_all_runners, get_runner

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
