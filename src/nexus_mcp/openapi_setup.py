"""OpenAPIProvider setup for auto-generating tools from OpenCode server spec.

Fetches the OpenAPI spec from /doc at startup, creates an OpenAPIProvider
with route maps that filter endpoints into workspace/monitoring/config tools,
and registers it on the FastMCP server.

All auto-generated tools share the authenticated httpx client from
OpenCodeHTTPClient.
"""

import logging

from fastmcp import FastMCP
from fastmcp.server.providers.openapi import MCPType, OpenAPIProvider, RouteMap

from nexus_mcp.http_client import OpenCodeHTTPClient

logger = logging.getLogger(__name__)

ROUTE_MAPS = [
    # Workspace (Phase 2)
    RouteMap(methods=["GET"], pattern=r"/find$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}),
    RouteMap(
        methods=["GET"], pattern=r"/find/file$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}
    ),
    RouteMap(
        methods=["GET"], pattern=r"/find/symbol$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}
    ),
    RouteMap(methods=["GET"], pattern=r"/file$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}),
    RouteMap(
        methods=["GET"], pattern=r"/file/content$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}
    ),
    RouteMap(
        methods=["GET"], pattern=r"/file/status$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}
    ),
    RouteMap(methods=["GET"], pattern=r"/vcs$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}),
    RouteMap(methods=["GET"], pattern=r"/project$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}),
    RouteMap(
        methods=["GET"], pattern=r"/project/current$", mcp_type=MCPType.TOOL, mcp_tags={"workspace"}
    ),
    # Monitoring (Phase 2)
    RouteMap(
        methods=["GET"], pattern=r"/global/health$", mcp_type=MCPType.TOOL, mcp_tags={"monitoring"}
    ),
    RouteMap(methods=["GET"], pattern=r"/lsp$", mcp_type=MCPType.TOOL, mcp_tags={"monitoring"}),
    RouteMap(methods=["GET"], pattern=r"/mcp$", mcp_type=MCPType.TOOL, mcp_tags={"monitoring"}),
    RouteMap(
        methods=["GET"], pattern=r"/formatter$", mcp_type=MCPType.TOOL, mcp_tags={"monitoring"}
    ),
    # Remaining config (Phase 2)
    RouteMap(
        methods=["GET"], pattern=r"/agent$", mcp_type=MCPType.TOOL, mcp_tags={"configuration"}
    ),
    RouteMap(
        methods=["GET"], pattern=r"/command$", mcp_type=MCPType.TOOL, mcp_tags={"configuration"}
    ),
    # Exclude everything else (catch-all — must be last)
    RouteMap(pattern=r".*", mcp_type=MCPType.EXCLUDE),
]


async def fetch_openapi_spec(client: OpenCodeHTTPClient | None = None) -> dict[str, object] | None:
    """Fetch the OpenAPI spec from the OpenCode server.

    Returns the spec as a dict, or None on failure.
    """
    if client is None:
        from nexus_mcp.http_client import get_http_client

        client = get_http_client()
    try:
        result = await client.get("/doc")
        if isinstance(result, dict) and "openapi" in result:
            return result
        logger.warning("OpenCode /doc returned unexpected format, skipping OpenAPIProvider")
        return None
    except Exception:
        logger.warning("Failed to fetch OpenAPI spec from /doc, skipping OpenAPIProvider")
        return None


async def setup_opencode_tools(mcp: FastMCP, client: OpenCodeHTTPClient) -> bool:
    """Set up OpenAPIProvider with auto-generated tools.

    Returns True if provider was successfully created and registered, False otherwise.
    """
    spec = await fetch_openapi_spec(client)
    if spec is None:
        return False

    try:
        provider = OpenAPIProvider(
            openapi_spec=spec,
            client=client._httpx,
            route_maps=ROUTE_MAPS,
        )
    except Exception as e:
        logger.warning("Failed to create OpenAPIProvider: %s", e)
        return False

    mcp.add_provider(provider)
    logger.info("OpenAPIProvider registered with auto-generated tools")
    return True
