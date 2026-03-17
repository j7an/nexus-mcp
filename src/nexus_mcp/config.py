"""Configuration module for nexus-mcp.

Provides a unified, tiered configuration system:
- OperationalDefaults: Pydantic model for the shape of all operational settings.
- NexusConfig: Top-level config with merged defaults + runner configs.
- HARDCODED_DEFAULTS: Lowest-priority baseline values.
- get_config(): Lazy singleton — reads env vars + TOML once on first access.
- reset_config(): Clear cached config for test isolation.
- get_runner_defaults(name): Per-runner merged OperationalDefaults.
- load_runner_config(): Backward-compatible TOML runner config accessor.
- Backward-compatible getter functions delegating to the singleton.

Resolution order (highest → lowest priority):
  per-request → session prefs → TOML [runner.X] → env vars → TOML [defaults] → hardcoded
"""

import math
import os
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from nexus_mcp.exceptions import ConfigurationError
from nexus_mcp.types import ExecutionMode

# ---------------------------------------------------------------------------
# Operational defaults shape
# ---------------------------------------------------------------------------


class OperationalDefaults(BaseModel, frozen=True):
    """Shape of operational settings at any tier.

    All fields are None-able — None means "not set at this tier".
    After merging all tiers, HARDCODED_DEFAULTS guarantees non-None for required fields.
    """

    timeout: int | None = Field(default=None, ge=1)
    output_limit: int | None = Field(default=None, ge=1)
    max_retries: int | None = Field(default=None, ge=1)
    retry_base_delay: Annotated[float, Field(ge=0)] | None = None
    retry_max_delay: Annotated[float, Field(ge=0)] | None = None
    tool_timeout: Annotated[float, Field(ge=0)] | None = None  # raw value; 0 → None in getter
    cli_detection_timeout: int | None = Field(default=None, ge=1)
    execution_mode: ExecutionMode | None = None
    model: str | None = Field(default=None, min_length=1)

    @field_validator("retry_base_delay", "retry_max_delay", "tool_timeout", mode="after")
    @classmethod
    def reject_non_finite(cls, v: float | None) -> float | None:
        """Safety net for programmatic construction. TOML rejects inf/nan at parse time.
        Env vars are validated manually in _read_env_defaults() with ConfigurationError."""
        if v is not None and not math.isfinite(v):
            raise ValueError(f"must be a finite number, got {v}")
        return v


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
# Runner config
# ---------------------------------------------------------------------------


class RunnerConfig(BaseModel, frozen=True):
    """Configuration for a runner from nexus-mcp.toml."""

    # Metadata fields
    type: Literal["cli", "server"] = "cli"
    provider: str | None = None
    models: tuple[str, ...] = ()
    url: str | None = None
    # Operational override fields (new — all optional, None means "use global default")
    timeout: int | None = None
    output_limit: int | None = None
    max_retries: int | None = None
    retry_base_delay: float | None = None
    retry_max_delay: float | None = None
    execution_mode: ExecutionMode | None = None
    model: str | None = None
    cli_path: str | None = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class NexusConfig(BaseModel, frozen=True):
    """Top-level config: merged global defaults + per-runner configs."""

    defaults: OperationalDefaults
    runners: dict[str, RunnerConfig]


# ---------------------------------------------------------------------------
# Environment variable parsing
# ---------------------------------------------------------------------------


def _read_env_defaults() -> OperationalDefaults:
    """Read operational defaults from environment variables.

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

    return OperationalDefaults(**kwargs)


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------


def _load_toml_config() -> tuple[OperationalDefaults, dict[str, RunnerConfig]]:
    """Load config from TOML file. Returns (toml_defaults, runners).

    Resolution order:
        1. NEXUS_CONFIG_PATH env var (absolute path)
        2. nexus-mcp.toml in current working directory

    Raises:
        ConfigurationError: If TOML is invalid or field types don't match.
    """
    config_path_env = os.getenv("NEXUS_CONFIG_PATH")
    path = Path(config_path_env if config_path_env is not None else "nexus-mcp.toml")

    if not path.is_file():
        return OperationalDefaults(), {}

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        msg = f"Invalid TOML in {path}: {e}"
        raise ConfigurationError(msg, config_key=str(path)) from e

    # Parse [defaults] section
    toml_defaults = OperationalDefaults()
    raw_defaults = data.get("defaults", {})
    if raw_defaults:
        try:
            toml_defaults = OperationalDefaults(**raw_defaults)
        except ValidationError as e:
            raise ConfigurationError(
                f"Invalid defaults config in {path}: {e}", config_key="defaults"
            ) from e

    # Parse [runner.*] sections
    runners_data = data.get("runner", {})
    runners: dict[str, RunnerConfig] = {}
    for name, cfg in runners_data.items():
        if not isinstance(cfg, dict):
            msg = f"Runner '{name}' must be a TOML table, got {type(cfg).__name__}"
            raise ConfigurationError(msg, config_key=f"runner.{name}")
        try:
            runners[name] = RunnerConfig(**cfg)
        except ValidationError as e:
            msg = f"Invalid runner config for '{name}' in {path}: {e}"
            raise ConfigurationError(msg, config_key=f"runner.{name}") from e

    return toml_defaults, runners


# ---------------------------------------------------------------------------
# Config singleton
# ---------------------------------------------------------------------------

_config: NexusConfig | None = None


def _load_config() -> NexusConfig:
    """Assemble NexusConfig: HARDCODED_DEFAULTS ← TOML [defaults] ← env vars."""
    env_defaults = _read_env_defaults()
    toml_defaults, runners = _load_toml_config()
    merged = _merge_defaults(HARDCODED_DEFAULTS, toml_defaults, env_defaults)
    return NexusConfig(defaults=merged, runners=runners)


def get_config() -> NexusConfig:
    """Lazy singleton. Reads env vars + TOML once on first access.

    For test isolation: call reset_config() before each test that patches env vars,
    then access the singleton within the test body.
    """
    global _config
    if _config is None:
        _config = _load_config()
    return _config


def reset_config() -> None:
    """Clear cached config. For test isolation — pair with RunnerFactory.clear_cache()."""
    global _config
    _config = None


# ---------------------------------------------------------------------------
# Per-runner defaults
# ---------------------------------------------------------------------------


def get_runner_defaults(runner_name: str) -> OperationalDefaults:
    """Get merged operational defaults for a specific runner.

    Merge chain: global config.defaults ← TOML [runner.X] fields ← per-agent env vars.

    Note: cli_path is NOT in OperationalDefaults (security-sensitive). Callers that need
    it must use get_agent_env(name, "PATH", default=name) or name directly.
    """
    config = get_config()
    runner_cfg = config.runners.get(runner_name)

    # Extract operational fields from RunnerConfig (exclude metadata-only fields)
    runner_overrides = OperationalDefaults()
    if runner_cfg is not None:
        runner_kwargs: dict[str, Any] = {}
        for field in (
            "timeout",
            "output_limit",
            "max_retries",
            "retry_base_delay",
            "retry_max_delay",
            "execution_mode",
            "model",
        ):
            value = getattr(runner_cfg, field)
            if value is not None:
                runner_kwargs[field] = value
        if runner_kwargs:
            runner_overrides = OperationalDefaults(**runner_kwargs)

    # Per-agent model env var is the highest-priority override for model
    agent_model = get_agent_env(runner_name, "MODEL")
    agent_overrides = (
        OperationalDefaults(model=agent_model) if agent_model is not None else OperationalDefaults()
    )

    return _merge_defaults(config.defaults, runner_overrides, agent_overrides)


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
# Backward-compatible getter functions (delegate to singleton)
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
    return get_config().defaults.output_limit  # type: ignore[return-value]


def get_global_timeout() -> int:
    """Get subprocess timeout in seconds.

    Returns:
        Timeout in seconds (default: 600s = 10 minutes)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_TIMEOUT_SECONDS: Subprocess timeout in seconds
    """
    return get_config().defaults.timeout  # type: ignore[return-value]


def get_retry_max_attempts() -> int:
    """Get maximum retry attempts.

    Returns:
        Max retry attempts (default: 3)

    Raises:
        ConfigurationError: If env var value is not a valid integer or not positive

    Environment Variable:
        NEXUS_RETRY_MAX_ATTEMPTS: Max number of attempts (including the first)
    """
    return get_config().defaults.max_retries  # type: ignore[return-value]


def get_retry_base_delay() -> float:
    """Get retry base delay in seconds.

    Returns:
        Base delay in seconds for exponential backoff (default: 2.0s)

    Raises:
        ConfigurationError: If env var value is not a valid float or is negative

    Environment Variable:
        NEXUS_RETRY_BASE_DELAY: Base seconds for exponential backoff
    """
    return get_config().defaults.retry_base_delay  # type: ignore[return-value]


def get_retry_max_delay() -> float:
    """Get retry max delay cap in seconds.

    Returns:
        Maximum delay cap in seconds (default: 60.0s)

    Raises:
        ConfigurationError: If env var value is not a valid float or is negative

    Environment Variable:
        NEXUS_RETRY_MAX_DELAY: Maximum seconds to wait between retries
    """
    return get_config().defaults.retry_max_delay  # type: ignore[return-value]


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
    value = get_config().defaults.tool_timeout
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
    return get_config().defaults.cli_detection_timeout  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Backward-compatible TOML runner config accessor
# ---------------------------------------------------------------------------


def load_runner_config() -> dict[str, RunnerConfig]:
    """Load runner configuration from nexus-mcp.toml.

    Delegates to get_config().runners. For test isolation, reset_config() before
    accessing with a custom NEXUS_CONFIG_PATH.

    Returns:
        Mapping of runner name to RunnerConfig. Empty dict if file not found.

    Raises:
        ConfigurationError: If TOML is invalid or field types don't match.
    """
    return get_config().runners
