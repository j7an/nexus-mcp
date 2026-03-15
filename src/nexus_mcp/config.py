"""Configuration module for environment variable settings (Tier 2).

Provides functions to read global and agent-specific configuration from
environment variables:
- Global: timeouts, output limits
- Agent-specific: CLI paths, default models
"""

import math
import os

from nexus_mcp.exceptions import ConfigurationError


def _get_int_env(env_var: str, default: str, label: str) -> int:
    raw = os.getenv(env_var, default)
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(f"Invalid {label} value: {raw!r}", config_key=env_var) from None


def _get_float_env(env_var: str, default: str, label: str) -> float:
    raw = os.getenv(env_var, default)
    try:
        value = float(raw)
    except ValueError:
        raise ConfigurationError(f"Invalid {label} value: {raw!r}", config_key=env_var) from None
    if not math.isfinite(value):
        raise ConfigurationError(
            f"{label.capitalize()} must be a finite number, got {value}",
            config_key=env_var,
        )
    return value


def get_global_output_limit() -> int:
    """Get maximum output size in bytes from env var.

    Returns:
        Output limit in bytes (default: 50KB)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_OUTPUT_LIMIT_BYTES: Max output size in bytes
    """
    value = _get_int_env("NEXUS_OUTPUT_LIMIT_BYTES", "50000", "output limit")
    if value <= 0:
        raise ConfigurationError(
            f"Output limit must be positive, got {value}",
            config_key="NEXUS_OUTPUT_LIMIT_BYTES",
        )
    return value


def get_global_timeout() -> int:
    """Get subprocess timeout in seconds from env var.

    Returns:
        Timeout in seconds (default: 600s = 10 minutes, matches process.py)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_TIMEOUT_SECONDS: Subprocess timeout in seconds
    """
    value = _get_int_env("NEXUS_TIMEOUT_SECONDS", "600", "timeout")
    if value <= 0:
        raise ConfigurationError(
            f"Timeout must be positive, got {value}",
            config_key="NEXUS_TIMEOUT_SECONDS",
        )
    return value


def get_retry_max_attempts() -> int:
    """Get maximum retry attempts from env var.

    Returns:
        Max retry attempts (default: 3)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_RETRY_MAX_ATTEMPTS: Max number of attempts (including the first)
    """
    value = _get_int_env("NEXUS_RETRY_MAX_ATTEMPTS", "3", "retry max attempts")
    if value <= 0:
        raise ConfigurationError(
            f"Retry max attempts must be positive, got {value}",
            config_key="NEXUS_RETRY_MAX_ATTEMPTS",
        )
    return value


def get_retry_base_delay() -> float:
    """Get retry base delay in seconds from env var.

    Returns:
        Base delay in seconds for exponential backoff (default: 2.0s)

    Raises:
        ConfigurationError: If env var value is not a valid float or is negative

    Environment Variable:
        NEXUS_RETRY_BASE_DELAY: Base seconds for exponential backoff
    """
    value = _get_float_env("NEXUS_RETRY_BASE_DELAY", "2.0", "retry base delay")
    if value < 0:
        raise ConfigurationError(
            f"Retry base delay must be non-negative, got {value}",
            config_key="NEXUS_RETRY_BASE_DELAY",
        )
    return value


def get_retry_max_delay() -> float:
    """Get retry max delay cap in seconds from env var.

    Returns:
        Maximum delay cap in seconds (default: 60.0s)

    Raises:
        ConfigurationError: If env var value is not a valid float or is negative

    Environment Variable:
        NEXUS_RETRY_MAX_DELAY: Maximum seconds to wait between retries
    """
    value = _get_float_env("NEXUS_RETRY_MAX_DELAY", "60.0", "retry max delay")
    if value < 0:
        raise ConfigurationError(
            f"Retry max delay must be non-negative, got {value}",
            config_key="NEXUS_RETRY_MAX_DELAY",
        )
    return value


def get_tool_timeout() -> float | None:
    """Get MCP tool-level timeout in seconds from env var.

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
    value = _get_float_env("NEXUS_TOOL_TIMEOUT_SECONDS", "900.0", "tool timeout")
    if value < 0:
        raise ConfigurationError(
            f"Tool timeout must be non-negative, got {value}",
            config_key="NEXUS_TOOL_TIMEOUT_SECONDS",
        )
    return value if value > 0 else None


def get_cli_detection_timeout() -> int:
    """Get CLI detection timeout in seconds from env var.

    Returns:
        Timeout in seconds (default: 30s)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_CLI_DETECTION_TIMEOUT: Seconds to wait for '<cli> --version'
    """
    value = _get_int_env("NEXUS_CLI_DETECTION_TIMEOUT", "30", "CLI detection timeout")
    if value <= 0:
        raise ConfigurationError(
            f"CLI detection timeout must be positive, got {value}",
            config_key="NEXUS_CLI_DETECTION_TIMEOUT",
        )
    return value


def get_agent_env(agent: str, key: str, default: str | None = None) -> str | None:
    """Get agent-specific environment variable.

    Args:
        agent: Agent name (e.g., "gemini", "codex")
        key: Config key (e.g., "PATH", "MODEL")
        default: Default value if env var not set

    Returns:
        Environment variable value or default

    Environment Variable Format:
        NEXUS_{AGENT}_{KEY} (e.g., NEXUS_GEMINI_PATH, NEXUS_CODEX_MODEL)

    Example:
        >>> get_agent_env("gemini", "PATH")  # Reads NEXUS_GEMINI_PATH
        "/usr/local/bin/gemini"
    """
    env_var = f"NEXUS_{agent.upper()}_{key}"
    return os.getenv(env_var, default)
