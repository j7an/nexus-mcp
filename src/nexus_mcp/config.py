"""Configuration module for nexus-mcp.

Provides a stateless, tiered configuration system:
- OperationalDefaults: Pydantic model for the shape of all operational settings.
- HARDCODED_DEFAULTS: Lowest-priority baseline values.
- get_runner_defaults(name): Per-runner merged OperationalDefaults.
- get_runner_models(name): Per-runner model list from env var.
- Backward-compatible getter functions reading env vars fresh each call.

Resolution order (highest → lowest priority):
  per-request → session prefs → per-runner env → global env → hardcoded
"""

import contextlib
import math
import os
from typing import Any

from nexus_mcp.exceptions import ConfigurationError
from nexus_mcp.types import OperationalDefaults

HARDCODED_DEFAULTS = OperationalDefaults(
    timeout=600,
    output_limit=50000,
    max_retries=3,
    retry_base_delay=2.0,
    retry_max_delay=60.0,
    tool_timeout=900.0,
    cli_detection_timeout=30,
    execution_mode="default",
)


def _merge_defaults(
    base: OperationalDefaults, *overlays: OperationalDefaults
) -> OperationalDefaults:
    """Merge operational defaults: non-None values from later overlays win."""
    result = base.model_dump()
    for overlay in overlays:
        for field, value in overlay.model_dump().items():
            if value is not None:
                result[field] = value
    return OperationalDefaults(**result)


# ---------------------------------------------------------------------------
# Environment variable parsing — global
# ---------------------------------------------------------------------------


def _read_global_env_defaults() -> OperationalDefaults:
    """Read operational defaults from global environment variables.

    Only populates fields where the corresponding env var is set.
    Preserves exact error messages and config_key values expected by existing tests.
    """
    kwargs: dict[str, Any] = {}

    for env_var, field, label in (
        ("NEXUS_TIMEOUT_SECONDS", "timeout", "timeout"),
        ("NEXUS_OUTPUT_LIMIT_BYTES", "output_limit", "output limit"),
        ("NEXUS_RETRY_MAX_ATTEMPTS", "max_retries", "retry max attempts"),
        ("NEXUS_CLI_DETECTION_TIMEOUT", "cli_detection_timeout", "CLI detection timeout"),
    ):
        raw = os.getenv(env_var)
        if raw is None:
            continue
        try:
            v = int(raw)
        except ValueError:
            raise ConfigurationError(
                f"Invalid {label} value: {raw!r}", config_key=env_var
            ) from None
        if v <= 0:
            raise ConfigurationError(
                f"{label.capitalize()} must be positive, got {v}", config_key=env_var
            )
        kwargs[field] = v

    for env_var, field, label in (
        ("NEXUS_RETRY_BASE_DELAY", "retry_base_delay", "retry base delay"),
        ("NEXUS_RETRY_MAX_DELAY", "retry_max_delay", "retry max delay"),
        ("NEXUS_TOOL_TIMEOUT_SECONDS", "tool_timeout", "tool timeout"),
    ):
        raw = os.getenv(env_var)
        if raw is None:
            continue
        try:
            fv = float(raw)
        except ValueError:
            raise ConfigurationError(
                f"Invalid {label} value: {raw!r}", config_key=env_var
            ) from None
        if not math.isfinite(fv):
            raise ConfigurationError(
                f"{label.capitalize()} must be a finite number, got {fv}", config_key=env_var
            )
        if fv < 0:
            raise ConfigurationError(
                f"{label.capitalize()} must be non-negative, got {fv}", config_key=env_var
            )
        kwargs[field] = fv

    # NEXUS_EXECUTION_MODE: global execution mode override
    raw_mode = os.getenv("NEXUS_EXECUTION_MODE")
    if raw_mode is not None:
        if raw_mode not in ("default", "yolo"):
            raise ConfigurationError(
                f"Invalid execution mode: {raw_mode!r} (must be 'default' or 'yolo')",
                config_key="NEXUS_EXECUTION_MODE",
            )
        kwargs["execution_mode"] = raw_mode

    return OperationalDefaults(**kwargs)


# ---------------------------------------------------------------------------
# Environment variable parsing — per-runner
# ---------------------------------------------------------------------------


def _read_runner_env_defaults(runner_name: str) -> OperationalDefaults:
    """Read per-runner operational defaults from NEXUS_{RUNNER}_{KEY} env vars.

    Only populates fields where the corresponding env var is set.
    E.g., for runner_name="gemini": NEXUS_GEMINI_TIMEOUT, NEXUS_GEMINI_MODEL, etc.
    """
    kwargs: dict[str, Any] = {}
    prefix = runner_name.upper()

    for key, field, parse in (
        ("TIMEOUT", "timeout", int),
        ("OUTPUT_LIMIT", "output_limit", int),
        ("MAX_RETRIES", "max_retries", int),
    ):
        raw = os.getenv(f"NEXUS_{prefix}_{key}")
        if raw is not None:
            with contextlib.suppress(ValueError):
                v = parse(raw)
                if v > 0:
                    kwargs[field] = v

    for key, field in (
        ("RETRY_BASE_DELAY", "retry_base_delay"),
        ("RETRY_MAX_DELAY", "retry_max_delay"),
    ):
        raw = os.getenv(f"NEXUS_{prefix}_{key}")
        if raw is not None:
            try:
                fv = float(raw)
                if math.isfinite(fv) and fv >= 0:
                    kwargs[field] = fv
            except ValueError:
                pass  # silently skip invalid per-runner overrides

    # Per-runner model: NEXUS_{RUNNER}_MODEL
    model = os.getenv(f"NEXUS_{prefix}_MODEL")
    if model:
        kwargs["model"] = model

    # Per-runner execution mode: NEXUS_{RUNNER}_EXECUTION_MODE
    mode = os.getenv(f"NEXUS_{prefix}_EXECUTION_MODE")
    if mode in ("default", "yolo"):
        kwargs["execution_mode"] = mode

    return OperationalDefaults(**kwargs)


# ---------------------------------------------------------------------------
# Per-runner models
# ---------------------------------------------------------------------------


def get_runner_models(runner_name: str) -> tuple[str, ...]:
    """Get model list for a runner from NEXUS_{RUNNER}_MODELS env var.

    The env var is comma-separated. Whitespace around each model name is stripped.
    Empty strings are filtered out.

    Returns:
        Tuple of model name strings, or empty tuple if env var not set.
    """
    raw = os.getenv(f"NEXUS_{runner_name.upper()}_MODELS")
    if not raw:
        return ()
    return tuple(m.strip() for m in raw.split(",") if m.strip())


# ---------------------------------------------------------------------------
# Per-runner defaults (3-tier merge)
# ---------------------------------------------------------------------------


def get_runner_defaults(runner_name: str) -> OperationalDefaults:
    """Get merged operational defaults for a specific runner.

    Merge chain: hardcoded → global env → per-runner env.
    """
    global_env = _read_global_env_defaults()
    runner_env = _read_runner_env_defaults(runner_name)
    return _merge_defaults(HARDCODED_DEFAULTS, global_env, runner_env)


# ---------------------------------------------------------------------------
# Agent env var helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Merged global defaults (stateless — reads env fresh each call)
# ---------------------------------------------------------------------------


def _get_merged_defaults() -> OperationalDefaults:
    """Return HARDCODED_DEFAULTS merged with global env var overrides.

    Used by backward-compatible getter functions. Reads env vars fresh each call.
    """
    return _merge_defaults(HARDCODED_DEFAULTS, _read_global_env_defaults())


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
