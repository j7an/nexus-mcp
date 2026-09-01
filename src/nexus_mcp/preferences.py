"""Compatibility imports for MCP preference tools."""

from nexus_mcp.mcp.preferences import (
    _apply_preferences,  # noqa: F401 - compatibility import
    _get_session_preferences,  # noqa: F401 - compatibility import
    _resolve_field,  # noqa: F401 - compatibility import
    clear_preferences,
    get_preferences,
    set_preferences,
)

__all__ = ["clear_preferences", "get_preferences", "set_preferences"]
