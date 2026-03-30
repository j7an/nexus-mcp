"""MCP resources for OpenCode server integration.

Exposes read-only resources for OpenCode server status, providers,
auth methods, and configuration. These replace the former read-only
tools following the 'tools mutate, resources read' pattern.

Resources:
- nexus://opencode: Integration status + tool group overview (always registered)
- nexus://opencode/providers: Provider list (conditional on server availability)
- nexus://opencode/providers/auth: Auth methods (conditional)
- nexus://opencode/config: Server configuration (conditional)
"""

import json
import logging
import os

from fastmcp import FastMCP

from nexus_mcp.config_resolver import get_opencode_server_url
from nexus_mcp.http_client import get_http_client

logger = logging.getLogger(__name__)

_RESOURCE_ANNOTATIONS = {"readOnlyHint": True, "idempotentHint": True}

_TOOL_GROUPS_HEALTHY = [
    {
        "tag": "configuration",
        "description": "Provider setup, config management",
        "tool_count": 5,
        "enabled": True,
    },
    {
        "tag": "workspace",
        "description": "File search, git status, project info",
        "tool_count": 9,
        "enabled": True,
    },
    {
        "tag": "monitoring",
        "description": "Server health, LSP/MCP status",
        "tool_count": 4,
        "enabled": True,
    },
    {
        "tag": "terminal",
        "description": "Shell commands, PTY sessions",
        "tool_count": 0,
        "enabled": False,
    },
]

_COMPOUND_TOOLS = ["opencode_investigate", "opencode_session_review"]


def is_opencode_server_configured() -> bool:
    """Check if the OpenCode server is configured via env vars."""
    return os.environ.get("NEXUS_OPENCODE_SERVER_PASSWORD") is not None


async def get_opencode_status() -> str:
    """Return OpenCode server integration status.

    Resource URI: nexus://opencode

    Always registered — reports configured=false when server is not set up,
    so client agents can understand why OpenCode tools are absent.
    """
    if not is_opencode_server_configured():
        return json.dumps(
            {
                "server": {"configured": False, "healthy": False, "url": None},
                "tool_groups": [],
                "compound_tools": [],
            }
        )

    url = get_opencode_server_url()
    client = get_http_client()
    try:
        healthy = await client.health_check()
    except Exception:
        logger.debug("Health check failed unexpectedly", exc_info=True)
        healthy = False

    return json.dumps(
        {
            "server": {"configured": True, "healthy": healthy, "url": url},
            "tool_groups": _TOOL_GROUPS_HEALTHY if healthy else [],
            "compound_tools": _COMPOUND_TOOLS if healthy else [],
        }
    )


async def get_opencode_providers() -> str:
    """Return provider list from OpenCode server.

    Resource URI: nexus://opencode/providers
    """
    data = await get_http_client().get("/provider")
    return json.dumps(data, indent=2)


async def get_opencode_providers_auth() -> str:
    """Return auth methods from OpenCode server.

    Resource URI: nexus://opencode/providers/auth
    """
    data = await get_http_client().get("/provider/auth")
    return json.dumps(data, indent=2)


async def get_opencode_config() -> str:
    """Return configuration from OpenCode server.

    Resource URI: nexus://opencode/config
    """
    data = await get_http_client().get("/config")
    return json.dumps(data, indent=2)


def register_opencode_status_resource(mcp: FastMCP) -> None:
    """Register the nexus://opencode status resource (always)."""
    mcp.resource(
        "nexus://opencode",
        mime_type="application/json",
        annotations=_RESOURCE_ANNOTATIONS,
    )(get_opencode_status)


def register_opencode_data_resources(mcp: FastMCP) -> None:
    """Register OpenCode data resources (when server is configured + healthy)."""
    mcp.resource(
        "nexus://opencode/providers",
        mime_type="application/json",
        annotations=_RESOURCE_ANNOTATIONS,
    )(get_opencode_providers)
    mcp.resource(
        "nexus://opencode/providers/auth",
        mime_type="application/json",
        annotations=_RESOURCE_ANNOTATIONS,
    )(get_opencode_providers_auth)
    mcp.resource(
        "nexus://opencode/config",
        mime_type="application/json",
        annotations=_RESOURCE_ANNOTATIONS,
    )(get_opencode_config)
