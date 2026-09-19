"""E2E: the registered tool/resource set never depends on OpenCode state."""

import httpx
import pytest
import respx
from fastmcp import Client
from fastmcp.exceptions import ToolError

from nexus_mcp.http_client import reset_http_client
from nexus_mcp.server import mcp

OPENCODE_TOOLS = {
    "opencode_set_provider_auth",
    "opencode_update_config",
    "opencode_investigate",
    "opencode_session_review",
}


@pytest.fixture(autouse=True)
def _mock_cli(mock_cli_detection):
    yield mock_cli_detection


async def _surface() -> tuple[set[str], set[str], set[str]]:
    reset_http_client()
    try:
        async with Client(mcp) as client:
            tools = {tool.name for tool in await client.list_tools()}
            resources = {str(resource.uri) for resource in await client.list_resources()}
            templates = {
                template.uriTemplate for template in await client.list_resource_templates()
            }
            return tools, resources, templates
    finally:
        mcp._lifespan_result_set = False
        reset_http_client()


@pytest.mark.e2e
async def test_surface_identical_across_opencode_states(monkeypatch):
    monkeypatch.delenv("NEXUS_OPENCODE_SERVER_PASSWORD", raising=False)
    unconfigured = await _surface()

    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "pw")
    with respx.mock(base_url="http://test:4096") as router:
        router.get("/global/health").mock(return_value=httpx.Response(503))
        unhealthy = await _surface()
    with respx.mock(base_url="http://test:4096") as router:
        router.get("/global/health").mock(return_value=httpx.Response(200, json={"healthy": True}))
        healthy = await _surface()

    assert unconfigured == unhealthy == healthy
    assert unconfigured[0] >= OPENCODE_TOOLS
    assert "nexus://opencode/providers" in unconfigured[1]


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("opencode_set_provider_auth", {"provider_id": "test", "credentials": {}}),
        ("opencode_update_config", {"config": {}}),
        ("opencode_investigate", {"query": "test"}),
        ("opencode_session_review", {"session_id": "ses_1"}),
    ],
)
async def test_opencode_tool_errors_clearly_when_unconfigured(
    monkeypatch, caplog, tool_name, arguments
):
    monkeypatch.delenv("NEXUS_OPENCODE_SERVER_PASSWORD", raising=False)
    reset_http_client()
    try:
        async with Client(mcp) as client:
            with (
                caplog.at_level("ERROR", logger="fastmcp.server.server"),
                pytest.raises(ToolError, match=r"^OpenCode server not configured"),
            ):
                await client.call_tool(tool_name, arguments)
            assert not any(
                record.exc_info
                for record in caplog.records
                if record.name == "fastmcp.server.server"
            )
    finally:
        mcp._lifespan_result_set = False
        reset_http_client()


@pytest.mark.e2e
async def test_opencode_data_resource_errors_clearly_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NEXUS_OPENCODE_SERVER_PASSWORD", raising=False)
    reset_http_client()
    try:
        async with Client(mcp) as client:
            with pytest.raises(Exception, match="OpenCode server not configured"):
                await client.read_resource("nexus://opencode/providers")
    finally:
        mcp._lifespan_result_set = False
        reset_http_client()
