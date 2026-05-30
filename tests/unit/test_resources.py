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
    "unclassified_models",
}


class TestGetAllRunners:
    """Tests for the nexus://runners resource."""

    async def test_returns_all_registered_runners(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        names = {r["name"] for r in result["runners"]}
        assert names == {"claude", "codex", "opencode", "opencode_server"}

    async def test_json_structure_has_required_keys(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        assert "runners" in result
        for runner in result["runners"]:
            assert set(runner.keys()) == RUNNER_INFO_KEYS

    async def test_installed_runner_has_path(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        codex = next(r for r in result["runners"] if r["name"] == "codex")
        assert codex["installed"] is True
        assert codex["path"] is not None

    async def test_supported_modes_match_runner_class(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        codex = next(r for r in result["runners"] if r["name"] == "codex")
        assert codex["supported_modes"] == ["default", "yolo"]
        opencode = next(r for r in result["runners"] if r["name"] == "opencode")
        assert opencode["supported_modes"] == ["default"]


class TestGetRunner:
    """Tests for the nexus://runners/{cli} template resource."""

    async def test_returns_correct_runner(self, mock_cli_detection):
        result = json.loads(await get_runner("codex"))
        assert result["name"] == "codex"
        assert set(result.keys()) == RUNNER_INFO_KEYS

    async def test_not_found_raises_resource_error(self, mock_cli_detection):
        with pytest.raises(ResourceError, match="unknown_cli"):
            await get_runner("unknown_cli")

    async def test_uninstalled_cli_returns_false_installed(self, mock_cli_detection):
        """When detect_cli returns found=False, installed/path/version are None/False."""
        from unittest.mock import patch

        from nexus_mcp.cli_detector import CLIInfo

        with patch("nexus_mcp.resources.detect_cli", return_value=CLIInfo(found=False)):
            result = json.loads(await get_runner("codex"))
        assert result["installed"] is False
        assert result["path"] is None
        assert result["version"] is None


class TestRunnerModelTierEnrichment:
    """Tests for model tier enrichment in runner resources."""

    async def test_models_are_objects_with_name_and_tier(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        for runner in result["runners"]:
            for model in runner["models"]:
                assert isinstance(model, dict)
                assert "name" in model
                assert "tier" in model
                assert model["tier"] in ("quick", "standard", "thorough")

    async def test_tier_matches_heuristic_when_no_saved_tiers(self, mock_cli_detection):
        from nexus_mcp.tiers import get_model_tier

        result = json.loads(await get_all_runners())
        codex = next(r for r in result["runners"] if r["name"] == "codex")
        for model_obj in codex["models"]:
            expected_tier = get_model_tier(model_obj["name"])
            assert model_obj["tier"] == expected_tier

    async def test_unclassified_models_listed(self, mock_cli_detection):
        result = json.loads(await get_all_runners())
        for runner in result["runners"]:
            assert "unclassified_models" in runner
            assert isinstance(runner["unclassified_models"], list)

    async def test_single_runner_has_enriched_models(self, mock_cli_detection):
        result = json.loads(await get_runner("codex"))
        for model in result["models"]:
            assert isinstance(model, dict)
            assert "name" in model
            assert "tier" in model

    @patch(
        "nexus_mcp.resources.get_runner_models",
        return_value=["gpt-5.4-mini", "gpt-5.4-pro"],
    )
    async def test_saved_tiers_override_heuristic(self, _mock_models, mock_cli_detection):
        """When saved tiers are provided, they take precedence over heuristics."""
        from nexus_mcp.resources import _build_runner_info

        # Without saved tiers, mini is "quick" by heuristic
        info_default = _build_runner_info("codex")
        mini_default = next(m for m in info_default["models"] if m["name"] == "gpt-5.4-mini")
        assert mini_default["tier"] == "quick"

        # With saved tiers, mini overridden to "thorough"
        saved = {"gpt-5.4-mini": "thorough"}
        info_saved = _build_runner_info("codex", saved_tiers=saved)
        mini_saved = next(m for m in info_saved["models"] if m["name"] == "gpt-5.4-mini")
        assert mini_saved["tier"] == "thorough"

    @patch(
        "nexus_mcp.resources.get_runner_models",
        return_value=["gpt-5.4-mini", "gpt-5.4-pro"],
    )
    async def test_saved_tier_model_not_in_unclassified(self, _mock_models, mock_cli_detection):
        """Models with saved tiers should NOT appear in unclassified_models."""
        from nexus_mcp.resources import _build_runner_info

        saved = {"gpt-5.4-mini": "quick", "gpt-5.4-pro": "thorough"}
        info = _build_runner_info("codex", saved_tiers=saved)
        for name in saved:
            assert name not in info["unclassified_models"]

    @patch("nexus_mcp.resources.load_model_tiers", side_effect=RuntimeError("store broken"))
    async def test_store_failure_falls_back_to_heuristics(self, mock_load, mock_cli_detection, ctx):
        """When load_model_tiers raises, resource still returns with heuristic tiers."""
        result = json.loads(await get_all_runners(ctx=ctx))
        assert "runners" in result
        for runner in result["runners"]:
            for model in runner["models"]:
                assert "tier" in model

    @patch(
        "nexus_mcp.resources.get_runner_models",
        return_value=["gpt-5.4-mini", "gpt-5.4-pro"],
    )
    @patch("nexus_mcp.resources.load_model_tiers")
    async def test_get_all_runners_with_ctx_loads_saved_tiers(
        self, mock_load, _mock_models, mock_cli_detection, ctx
    ):
        """When ctx is provided, saved tiers are loaded and used."""
        mock_load.return_value = {"gpt-5.4-mini": "thorough"}
        result = json.loads(await get_all_runners(ctx=ctx))
        mock_load.assert_awaited_once_with(ctx)
        codex = next(r for r in result["runners"] if r["name"] == "codex")
        mini = next(m for m in codex["models"] if m["name"] == "gpt-5.4-mini")
        assert mini["tier"] == "thorough"

    @patch(
        "nexus_mcp.resources.get_runner_models",
        return_value=["gpt-5.4-mini", "gpt-5.4-pro"],
    )
    @patch("nexus_mcp.resources.load_model_tiers")
    async def test_get_runner_with_ctx_loads_saved_tiers(
        self, mock_load, _mock_models, mock_cli_detection, ctx
    ):
        """get_runner() also uses saved tiers when ctx is provided."""
        mock_load.return_value = {"gpt-5.4-mini": "thorough"}
        result = json.loads(await get_runner("codex", ctx=ctx))
        mini = next(m for m in result["models"] if m["name"] == "gpt-5.4-mini")
        assert mini["tier"] == "thorough"


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
        mock_load.return_value = {"execution_mode": "yolo", "model": "gpt-5.4-pro"}
        result = json.loads(await get_preferences_resource(ctx=ctx))
        assert result["source"] == "session"
        assert result["preferences"]["execution_mode"] == "yolo"
        assert result["preferences"]["model"] == "gpt-5.4-pro"

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
