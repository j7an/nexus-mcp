"""Compatibility imports for OpenCode MCP resources."""

from nexus_mcp.mcp.opencode_resources import (
    _MESSAGE_ID_PATTERN,  # noqa: F401 - compatibility import
    _SESSION_ID_PATTERN,  # noqa: F401 - compatibility import
    get_opencode_config,
    get_opencode_permissions,
    get_opencode_providers,
    get_opencode_providers_auth,
    get_opencode_questions,
    get_opencode_sessions,
    get_opencode_sessions_status,
    get_opencode_status,
    get_session_children,
    get_session_diff,
    get_session_message,
    get_session_messages,
    get_session_todo,
    is_opencode_server_configured,
    register_opencode_data_resources,
    register_opencode_status_resource,
)

__all__ = [
    "get_opencode_config",
    "get_opencode_permissions",
    "get_opencode_providers",
    "get_opencode_providers_auth",
    "get_opencode_questions",
    "get_opencode_sessions",
    "get_opencode_sessions_status",
    "get_opencode_status",
    "get_session_children",
    "get_session_diff",
    "get_session_message",
    "get_session_messages",
    "get_session_todo",
    "is_opencode_server_configured",
    "register_opencode_data_resources",
    "register_opencode_status_resource",
]
