"""Configuration module for environment variable settings (Tier 2).

Provides functions to read global and agent-specific configuration from
environment variables:
- Global: timeouts, output limits
- Agent-specific: CLI paths, default models
"""

import os

from nexus_mcp.exceptions import ConfigurationError


def get_global_output_limit() -> int:
    """Get maximum output size in bytes from env var.

    Returns:
        Output limit in bytes (default: 50KB)

    Raises:
        ConfigurationError: If env var value is not a valid integer

    Environment Variable:
        NEXUS_OUTPUT_LIMIT_BYTES: Max output size in bytes
    """
    raw = os.getenv("NEXUS_OUTPUT_LIMIT_BYTES", "50000")
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(
            f"Invalid output limit value: {raw!r}", config_key="NEXUS_OUTPUT_LIMIT_BYTES"
        ) from None


def get_global_timeout() -> int:
    """Get subprocess timeout in seconds from env var.

    Returns:
        Timeout in seconds (default: 600s = 10 minutes, matches process.py)

    Raises:
        ConfigurationError: If env var value is not a valid integer

    Environment Variable:
        NEXUS_TIMEOUT_SECONDS: Subprocess timeout in seconds
    """
    raw = os.getenv("NEXUS_TIMEOUT_SECONDS", "600")
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(
            f"Invalid timeout value: {raw!r}", config_key="NEXUS_TIMEOUT_SECONDS"
        ) from None


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
