"""Compatibility imports for the FastMCP server implementation."""

from nexus_mcp.mcp.preferences import clear_preferences, set_preferences
from nexus_mcp.mcp.server import (
    _inject_cli_enum,  # noqa: F401 - compatibility import
    batch_prompt,
    build_server_instructions,
    mcp,
    opencode_set_provider_auth,  # noqa: F401 - compatibility import
    opencode_update_config,  # noqa: F401 - compatibility import
    prompt,
    set_model_tiers,  # noqa: F401 - compatibility import
)

__all__ = [
    "batch_prompt",
    "build_server_instructions",
    "clear_preferences",
    "mcp",
    "prompt",
    "set_preferences",
]
