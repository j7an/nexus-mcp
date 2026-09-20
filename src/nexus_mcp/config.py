"""Configuration module for nexus-mcp.

Public API for configuration access. Resolution logic lives in config_resolver.py.

Resolution order (highest → lowest priority):
  per-request → session prefs → per-runner env → global env → hardcoded
"""

import os
from typing import Literal, cast

from nexus_mcp.exceptions import ConfigurationError

__all__ = [
    "HARDCODED_DEFAULTS",
    "get_tool_timeout",
    "get_cli_detection_timeout",
    "get_runner_defaults",
    "get_runner_models",
    "get_agent_env",
    "get_legacy_runners_enabled",
    "get_claude_settings_profile",
    "ClaudeSettingsProfile",
]

# Re-export resolution functions and HARDCODED_DEFAULTS so existing imports
# continue to work. No circular import: config_resolver.py does NOT import
# from config.py. HARDCODED_DEFAULTS is defined in config_resolver.py.
from nexus_mcp.config_resolver import (
    HARDCODED_DEFAULTS as HARDCODED_DEFAULTS,
)
from nexus_mcp.config_resolver import (
    _get_merged_defaults as _get_merged_defaults,
)
from nexus_mcp.config_resolver import (
    _merge_defaults as _merge_defaults,
)
from nexus_mcp.config_resolver import (
    _read_global_env_defaults as _read_global_env_defaults,
)
from nexus_mcp.config_resolver import (
    _read_runner_env_defaults as _read_runner_env_defaults,
)
from nexus_mcp.config_resolver import (
    get_agent_env as get_agent_env,
)
from nexus_mcp.config_resolver import (
    get_runner_defaults as get_runner_defaults,
)
from nexus_mcp.config_resolver import (
    get_runner_models as get_runner_models,
)

# ---------------------------------------------------------------------------
# Backward-compatible getter functions
# ---------------------------------------------------------------------------


type ClaudeSettingsProfile = Literal["isolated", "project", "inherit"]

_CLAUDE_SETTINGS_PROFILES = ("isolated", "project", "inherit")


def get_legacy_runners_enabled() -> bool:
    """Return whether legacy CLI runners replace their native backends.

    Environment Variable:
        NEXUS_ENABLE_LEGACY_RUNNERS: "1" restores the legacy runner for every backend
            that has a native replacement. Any other value keeps native backends.
    """
    return os.environ.get("NEXUS_ENABLE_LEGACY_RUNNERS", "").strip() == "1"


def get_claude_settings_profile() -> ClaudeSettingsProfile:
    """Return which Claude settings sources the Claude Agent backend may load.

    Raises:
        ConfigurationError: If the env var names an unknown profile.

    Environment Variable:
        NEXUS_CLAUDE_SETTINGS_PROFILE: "isolated" (default), "project", or "inherit".
    """
    value = os.environ.get("NEXUS_CLAUDE_SETTINGS_PROFILE", "isolated").strip().lower()
    if value not in _CLAUDE_SETTINGS_PROFILES:
        raise ConfigurationError(
            f"NEXUS_CLAUDE_SETTINGS_PROFILE must be one of {_CLAUDE_SETTINGS_PROFILES}",
            config_key="NEXUS_CLAUDE_SETTINGS_PROFILE",
        )
    return cast("ClaudeSettingsProfile", value)


def get_tool_timeout() -> float | None:
    """Get MCP tool-level timeout in seconds.

    Returns:
        Timeout in seconds applied via anyio.fail_after() (default: 900s = 15 min),
        or None to disable (0 → None). Set above the subprocess timeout (600s) to
        catch runaway retry loops.

    Raises:
        ConfigurationError: If env var value is not a finite non-negative number

    Environment Variable:
        NEXUS_TOOL_TIMEOUT_SECONDS: Seconds before FastMCP cancels a hung tool call.
            Set to 0 to disable. Must be finite and non-negative.
    """
    value = _get_merged_defaults().tool_timeout
    return value if value and value > 0 else None


def get_cli_detection_timeout() -> int:
    """Get CLI detection timeout in seconds.

    Returns:
        Timeout in seconds (default: 30s)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_CLI_DETECTION_TIMEOUT: Seconds to wait for '<cli> --version'
    """
    return _get_merged_defaults().cli_detection_timeout  # type: ignore[return-value]
