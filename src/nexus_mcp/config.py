"""Configuration module for nexus-mcp.

Public API for configuration access. Resolution logic lives in config_resolver.py.

Resolution order (highest → lowest priority):
  per-request → session prefs → per-runner env → global env → hardcoded
"""

__all__ = [
    "HARDCODED_DEFAULTS",
    "get_global_output_limit",
    "get_global_timeout",
    "get_retry_max_attempts",
    "get_retry_base_delay",
    "get_retry_max_delay",
    "get_tool_timeout",
    "get_cli_detection_timeout",
    "get_runner_defaults",
    "get_runner_models",
    "get_agent_fallback_models",
    "get_agent_env",
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
    get_agent_fallback_models as get_agent_fallback_models,
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


def get_global_output_limit() -> int:
    """Get maximum output size in bytes.

    Returns:
        Output limit in bytes (default: 50KB)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_OUTPUT_LIMIT_BYTES: Max output size in bytes
    """
    return _get_merged_defaults().output_limit  # type: ignore[return-value]


def get_global_timeout() -> int:
    """Get subprocess timeout in seconds.

    Returns:
        Timeout in seconds (default: 600s = 10 minutes)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_TIMEOUT_SECONDS: Subprocess timeout in seconds
    """
    return _get_merged_defaults().timeout  # type: ignore[return-value]


def get_retry_max_attempts() -> int:
    """Get maximum retry attempts.

    Returns:
        Max retry attempts (default: 3)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_RETRY_MAX_ATTEMPTS: Max number of attempts (including the first)
    """
    return _get_merged_defaults().max_retries  # type: ignore[return-value]


def get_retry_base_delay() -> float:
    """Get retry base delay in seconds.

    Returns:
        Base delay in seconds for exponential backoff (default: 2.0s)

    Raises:
        ConfigurationError: If env var value is not a valid float or is negative

    Environment Variable:
        NEXUS_RETRY_BASE_DELAY: Base seconds for exponential backoff
    """
    return _get_merged_defaults().retry_base_delay  # type: ignore[return-value]


def get_retry_max_delay() -> float:
    """Get retry max delay cap in seconds.

    Returns:
        Maximum delay cap in seconds (default: 60.0s)

    Raises:
        ConfigurationError: If env var value is not a valid float or is negative

    Environment Variable:
        NEXUS_RETRY_MAX_DELAY: Maximum seconds to wait between retries
    """
    return _get_merged_defaults().retry_max_delay  # type: ignore[return-value]


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
