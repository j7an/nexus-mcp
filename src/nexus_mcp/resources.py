"""Compatibility imports for MCP resource implementations."""

from nexus_mcp.mcp.resources import (
    _build_runner_info,  # noqa: F401 - compatibility import
    _load_saved_tiers,  # noqa: F401 - compatibility import
    get_all_runners,
    get_config,
    get_preferences_resource,
    get_runner,
    get_tiers_resource,
    register_resources,
)

__all__ = [
    "get_all_runners",
    "get_config",
    "get_preferences_resource",
    "get_runner",
    "get_tiers_resource",
    "register_resources",
]
