"""Admission-time snapshots of credential-free Nexus execution configuration."""

import hashlib
import json
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nexus_mcp.core.models import (
    ApprovalPolicy,
    ConfigLayerSnapshot,
    ExecutionConfigValues,
    RequestedExecutionConfig,
    RetryPolicy,
    SandboxMode,
    Workspace,
)
from nexus_mcp.exceptions import ConfigurationError

__all__ = [
    "NexusConfigResolver",
    "TOMLConfigurationFile",
    "TOMLExecutionConfigValues",
    "read_config_file",
    "user_config_path",
    "workspace_config_path",
]


class TOMLExecutionConfigValues(BaseModel):
    """One closed TOML configuration section before retry fields are normalized."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    model: str | None = Field(default=None, min_length=1, max_length=256)
    reasoning_effort: str | None = Field(default=None, min_length=1, max_length=64)
    sandbox: SandboxMode | None = None
    approval_policy: ApprovalPolicy | None = None
    timeout_seconds: int | None = Field(default=None, ge=1)
    output_limit_bytes: int | None = Field(default=None, ge=1)
    max_attempts: int | None = Field(default=None, ge=1)
    retry_base_delay_seconds: float | None = Field(default=None, ge=0)
    retry_max_delay_seconds: float | None = Field(default=None, ge=0)

    def to_execution_values(self) -> ExecutionConfigValues:
        """Normalize one TOML section through the core execution configuration model."""
        values: dict[str, object] = self.model_dump(exclude_none=True)
        retry_values = {
            "max_attempts": values.pop("max_attempts", None),
            "base_delay_seconds": values.pop("retry_base_delay_seconds", None),
            "max_delay_seconds": values.pop("retry_max_delay_seconds", None),
        }
        present_retry_values = {
            key: value for key, value in retry_values.items() if value is not None
        }
        if present_retry_values:
            values["retry_policy"] = RetryPolicy.model_validate(present_retry_values)
        return ExecutionConfigValues.model_validate(values)


class TOMLConfigurationFile(BaseModel):
    """The complete, closed Nexus TOML document schema."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    defaults: TOMLExecutionConfigValues = Field(default_factory=TOMLExecutionConfigValues)
    backends: dict[str, TOMLExecutionConfigValues] = Field(default_factory=dict)

    def execution_values(self, backend_id: str) -> ExecutionConfigValues:
        """Apply one backend section over this file's defaults before core normalization."""
        backend_values = self.backends.get(backend_id)
        if backend_values is None:
            return self.defaults.to_execution_values()
        merged_values = self.defaults.model_dump(exclude_none=True) | backend_values.model_dump(
            exclude_none=True
        )
        return TOMLExecutionConfigValues.model_validate(merged_values).to_execution_values()


def user_config_path() -> Path:
    """Return the override or platform-standard user configuration file location."""
    if override := os.environ.get("NEXUS_CONFIG_PATH"):
        return Path(override).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "nexus-mcp" / "config.toml"


def workspace_config_path(workspace: Workspace) -> Path:
    """Return the fixed Nexus configuration location within a canonical workspace."""
    return workspace.canonical_path / ".nexus" / "config.toml"


def read_config_file(path: Path, *, backend_id: str) -> ExecutionConfigValues:
    """Read one strict TOML file and select its backend-adjusted values."""
    contents = _read_config_contents(path)
    return _parse_config_contents(contents, backend_id=backend_id, config_key=str(path))


@dataclass(frozen=True, slots=True)
class NexusConfigResolver:
    """Capture immutable Nexus-controlled layers at admission time."""

    user_path: Path | None = None

    def snapshot(
        self,
        backend_id: str,
        workspace: Workspace,
        explicit: ExecutionConfigValues,
    ) -> RequestedExecutionConfig:
        """Capture workspace, user, and environment layers without provider-native defaults."""
        workspace_snapshot = _file_snapshot(workspace_config_path(workspace), backend_id)
        user_path = (
            self.user_path.expanduser() if self.user_path is not None else user_config_path()
        )
        user_snapshot = _file_snapshot(user_path, backend_id)
        environment_snapshot = _environment_snapshot(backend_id)
        return RequestedExecutionConfig(
            explicit=explicit,
            workspace=workspace_snapshot,
            user=user_snapshot,
            environment=environment_snapshot,
        )


def _file_snapshot(path: Path, backend_id: str) -> ConfigLayerSnapshot | None:
    """Return one immutable file-backed layer when its configuration file exists."""
    if not path.exists():
        return None
    contents = _read_config_contents(path)
    values = _parse_config_contents(contents, backend_id=backend_id, config_key=str(path))
    return ConfigLayerSnapshot(
        values=values,
        source=str(path),
        source_hash=hashlib.sha256(contents).hexdigest(),
    )


def _read_config_contents(path: Path) -> bytes:
    """Read configuration bytes once so parsed values and content hash share one input."""
    try:
        return path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            "unable to read Nexus configuration", config_key=str(path)
        ) from error


def _parse_config_contents(
    contents: bytes,
    *,
    backend_id: str,
    config_key: str,
) -> ExecutionConfigValues:
    """Parse strict TOML without retaining its text or validation input in errors."""
    try:
        parsed: dict[str, Any] = tomllib.loads(contents.decode("utf-8"))
        configuration = TOMLConfigurationFile.model_validate(parsed)
        values = configuration.execution_values(backend_id)
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError):
        sanitized_error = ConfigurationError("invalid Nexus configuration", config_key=config_key)
    else:
        return values
    raise sanitized_error


def _environment_snapshot(backend_id: str) -> ConfigLayerSnapshot | None:
    """Capture the narrowly supported execution variables without recording their names."""
    values = _environment_values(backend_id)
    if values is None:
        return None
    normalized_values = values.model_dump(mode="json", exclude_none=True)
    encoded_values = json.dumps(normalized_values, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return ConfigLayerSnapshot(
        values=values,
        source="environment",
        source_hash=hashlib.sha256(encoded_values).hexdigest(),
    )


def _environment_values(backend_id: str) -> ExecutionConfigValues | None:
    """Build core configuration values from only Task 9's documented environment mapping."""
    values: dict[str, object] = {}
    source_keys: dict[str, str] = {}
    _set_environment_value(values, source_keys, "timeout_seconds", "NEXUS_TIMEOUT_SECONDS")
    _set_environment_value(values, source_keys, "output_limit_bytes", "NEXUS_OUTPUT_LIMIT_BYTES")
    retry_values: dict[str, str] = {}
    _set_environment_value(
        retry_values,
        source_keys,
        "max_attempts",
        "NEXUS_RETRY_MAX_ATTEMPTS",
        source_field="retry_policy.max_attempts",
    )
    _set_environment_value(
        retry_values,
        source_keys,
        "base_delay_seconds",
        "NEXUS_RETRY_BASE_DELAY",
        source_field="retry_policy.base_delay_seconds",
    )
    _set_environment_value(
        retry_values,
        source_keys,
        "max_delay_seconds",
        "NEXUS_RETRY_MAX_DELAY",
        source_field="retry_policy.max_delay_seconds",
    )
    if retry_values:
        values["retry_policy"] = retry_values
    _set_environment_value(values, source_keys, "model", f"NEXUS_{backend_id.upper()}_MODEL")
    if not values:
        return None
    try:
        return ExecutionConfigValues.model_validate(values)
    except ValidationError as error:
        location = error.errors()[0]["loc"]
        field_name = ".".join(str(part) for part in location)
        sanitized_error = ConfigurationError(
            "invalid Nexus environment configuration",
            config_key=source_keys.get(field_name),
        )
    raise sanitized_error


def _set_environment_value(
    values: dict[str, object] | dict[str, str],
    source_keys: dict[str, str],
    field_name: str,
    environment_name: str,
    *,
    source_field: str | None = None,
) -> None:
    """Copy one configured environment value without retaining its variable name in a snapshot."""
    value = os.environ.get(environment_name)
    if value is not None:
        values[field_name] = value
        source_keys[source_field or field_name] = environment_name
