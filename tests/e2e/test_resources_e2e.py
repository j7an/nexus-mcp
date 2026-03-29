# tests/e2e/test_resources_e2e.py
"""E2E MCP resource tests using FastMCP's in-process Client.

Tests the full MCP stack for resources:
- Resource discovery via list_resources() → JSON-RPC
- Resource reading via read_resource() → JSON-RPC
- URI template resolution for nexus://runners/{cli}
- ResourceError propagation for unknown CLIs
- Annotations on all resources

Mock boundary: CLI detection only (via autouse mock_cli_detection).
All layers above run for real, including JSON-RPC dispatch.
"""

import json

import pytest
from mcp.shared.exceptions import McpError


@pytest.mark.e2e
class TestResourceDiscovery:
    """Verify MCP resource registration via list_resources() JSON-RPC call."""

    async def test_list_resources_includes_all_static_uris(self, mcp_client):
        """list_resources() includes the 2 static resource URIs."""
        resources = await mcp_client.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "nexus://runners" in uris
        assert "nexus://config" in uris

    async def test_list_resources_includes_preferences(self, mcp_client):
        """list_resources() includes the preferences resource."""
        resources = await mcp_client.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "nexus://preferences" in uris

    async def test_all_resources_have_json_mime_type(self, mcp_client):
        """All registered resources have application/json MIME type."""
        resources = await mcp_client.list_resources()
        for resource in resources:
            assert resource.mimeType == "application/json", (
                f"Resource {resource.uri} has MIME type {resource.mimeType!r}"
            )


@pytest.mark.e2e
class TestResourceTemplateDiscovery:
    """Verify URI template resources appear in resource_templates list."""

    async def test_runner_template_listed(self, mcp_client):
        """nexus://runners/{cli} appears in list_resource_templates()."""
        templates = await mcp_client.list_resource_templates()
        template_uris = {str(t.uriTemplate) for t in templates}
        assert "nexus://runners/{cli}" in template_uris


@pytest.mark.e2e
class TestReadRunnersResource:
    """Verify nexus://runners resource via read_resource()."""

    async def test_returns_valid_json_with_all_runners(self, mcp_client):
        contents = await mcp_client.read_resource("nexus://runners")
        data = json.loads(contents[0].text)
        names = {r["name"] for r in data["runners"]}
        assert names == {"claude", "codex", "gemini", "opencode", "opencode_server"}

    async def test_each_runner_has_required_fields(self, mcp_client):
        contents = await mcp_client.read_resource("nexus://runners")
        data = json.loads(contents[0].text)
        expected_keys = {
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
        for runner in data["runners"]:
            assert set(runner.keys()) == expected_keys

    async def test_models_are_enriched_with_tier_data(self, mcp_client):
        contents = await mcp_client.read_resource("nexus://runners")
        data = json.loads(contents[0].text)
        for runner in data["runners"]:
            for model in runner["models"]:
                assert isinstance(model, dict)
                assert "name" in model
                assert "tier" in model
                assert model["tier"] in ("quick", "standard", "thorough")


@pytest.mark.e2e
class TestReadRunnerTemplate:
    """Verify nexus://runners/{cli} template resource via read_resource()."""

    async def test_template_resolves_for_known_cli(self, mcp_client):
        contents = await mcp_client.read_resource("nexus://runners/gemini")
        data = json.loads(contents[0].text)
        assert data["name"] == "gemini"
        assert isinstance(data["supported_modes"], list)

    async def test_template_returns_error_for_unknown_cli(self, mcp_client):
        with pytest.raises(McpError):
            await mcp_client.read_resource("nexus://runners/nonexistent")


@pytest.mark.e2e
class TestReadConfigResource:
    """Verify nexus://config resource via read_resource()."""

    async def test_returns_resolved_defaults(self, mcp_client):
        contents = await mcp_client.read_resource("nexus://config")
        data = json.loads(contents[0].text)
        assert "timeout" in data
        assert "max_retries" in data
        assert all(v is not None for v in data.values())


@pytest.mark.e2e
class TestReadPreferencesResource:
    """Verify nexus://preferences resource via read_resource()."""

    async def test_returns_preferences_with_source(self, mcp_client):
        contents = await mcp_client.read_resource("nexus://preferences")
        data = json.loads(contents[0].text)
        assert data["source"] in ("session", "defaults")
        assert "preferences" in data


@pytest.mark.e2e
class TestReadTiersResource:
    """Verify nexus://tiers resource via read_resource()."""

    async def test_returns_empty_dict_when_no_tiers_saved(self, mcp_client):
        contents = await mcp_client.read_resource("nexus://tiers")
        data = json.loads(contents[0].text)
        assert data == {}

    async def test_returns_saved_tiers(self, mcp_client):
        await mcp_client.call_tool(
            "set_model_tiers",
            {"tiers": {"gemini-2.5-flash": "quick", "kimi-k2.5": "standard"}},
        )
        contents = await mcp_client.read_resource("nexus://tiers")
        data = json.loads(contents[0].text)
        assert data == {"gemini-2.5-flash": "quick", "kimi-k2.5": "standard"}

    async def test_tiers_resource_listed_in_resources(self, mcp_client):
        resources = await mcp_client.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "nexus://tiers" in uris


@pytest.mark.e2e
class TestResourceAnnotations:
    """Verify all resources have readOnlyHint and idempotentHint annotations."""

    async def test_all_resources_have_annotations(self, mcp_client):
        """Every resource and template has readOnlyHint=True, idempotentHint=True."""
        resources = await mcp_client.list_resources()
        for resource in resources:
            assert resource.annotations is not None, f"Resource {resource.uri} missing annotations"
            assert resource.annotations.readOnlyHint is True
            assert resource.annotations.idempotentHint is True

    async def test_template_resources_have_annotations(self, mcp_client):
        templates = await mcp_client.list_resource_templates()
        for template in templates:
            assert template.annotations is not None, (
                f"Template {template.uriTemplate} missing annotations"
            )
            assert template.annotations.readOnlyHint is True
            assert template.annotations.idempotentHint is True
