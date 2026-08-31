"""Trusted local access-context construction for the MCP adapter."""

import getpass
import os
import platform

from nexus_mcp.core import AccessContext

__all__ = ["local_access_context", "local_principal_id"]


def local_principal_id() -> str:
    """Return the stable local operating-system identity for this MCP process."""
    if hasattr(os, "getuid"):
        return f"local:{os.getuid()}"
    account = getpass.getuser().strip().casefold()
    machine = platform.node().strip().casefold()
    return f"local-windows:{machine}:{account}"


def local_access_context() -> AccessContext:
    """Build the adapter-owned local trust boundary for durable job access."""
    return AccessContext(
        principal_id=local_principal_id(),
        authentication_kind="local",
        authorize_local_workspaces=True,
    )
