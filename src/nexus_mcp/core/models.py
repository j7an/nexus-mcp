"""Framework-independent immutable primitives for Nexus jobs and configuration."""

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from nexus_mcp.core.errors import InvalidJobTransitionError
from nexus_mcp.core.operations import AgentOperation, OperationKind

if TYPE_CHECKING:
    from nexus_mcp.core.interaction import PendingInput

__all__ = [
    "ALLOWED_TRANSITIONS",
    "TERMINAL_STATES",
    "AccessContext",
    "AccessPolicy",
    "AgentJob",
    "AgentSession",
    "ApprovalPolicy",
    "BackendEvent",
    "ConfigLayerSnapshot",
    "ConfigSource",
    "ExecutionConfigValues",
    "FallbackConfigSource",
    "JobAttempt",
    "JobEvent",
    "JobEventType",
    "JobHandle",
    "JobListPage",
    "JobPhase",
    "JobState",
    "JobStatus",
    "ProviderReference",
    "RequestedExecutionConfig",
    "ResolvedExecutionConfig",
    "RetryPolicy",
    "SandboxMode",
    "Workspace",
    "WorkspaceSelector",
    "new_id",
    "validate_job_transition",
]

type JobState = Literal["queued", "running", "input_required", "completed", "failed", "cancelled"]
type JobPhase = Literal[
    "claiming", "starting", "executing", "reconciling", "interrupting", "finalizing"
]
type AccessPolicy = Literal["private", "workspace"]
type SandboxMode = Literal["read_only", "workspace_write", "danger_full_access"]
type ApprovalPolicy = Literal["provider_default", "on_request", "never"]
type JobEventType = Literal[
    "job_queued",
    "job_started",
    "progress",
    "provider_connected",
    "provider_disconnected",
    "reconciliation",
    "log",
    "command",
    "file_change",
    "input_required",
    "input_resolved",
    "message",
    "usage",
    "cancel_requested",
    "retry_scheduled",
    "job_completed",
    "job_failed",
    "job_cancelled",
]
type FallbackConfigSource = Literal["fallback", "legacy_nexus_fallback"]
type ConfigSource = Literal[
    "explicit",
    "provider",
    "workspace",
    "user",
    "environment",
    "fallback",
    "legacy_nexus_fallback",
]
type ConfigFieldName = Literal[
    "model",
    "reasoning_effort",
    "sandbox",
    "approval_policy",
    "timeout_seconds",
    "output_limit_bytes",
    "retry_policy",
]
type ConfigValue = str | int | "RetryPolicy"

TERMINAL_STATES: frozenset[JobState] = frozenset({"completed", "failed", "cancelled"})
ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    "queued": frozenset({"running", "cancelled"}),
    "running": frozenset({"input_required", "completed", "failed", "cancelled"}),
    "input_required": frozenset({"running", "completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be a timezone-aware UTC datetime")
    return value.astimezone(UTC)


def _freeze_json_value(value: JsonValue) -> JsonValue:
    """Recursively replace mutable JSON containers with read-only equivalents."""
    if isinstance(value, dict):
        frozen = MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
        return cast("JsonValue", frozen)
    if isinstance(value, list):
        return cast("JsonValue", tuple(_freeze_json_value(item) for item in value))
    return value


def _freeze_json_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})


def _thaw_json_value(value: JsonValue) -> JsonValue:
    """Restore ordinary JSON containers at the serialization boundary."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(cast("JsonValue", item)) for item in value]
    return value


def _thaw_json_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: _thaw_json_value(item) for key, item in value.items()}


class FrozenModel(BaseModel):
    """Shared Pydantic settings for persisted, closed domain values."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RetryPolicy(FrozenModel):
    """Bounded retry settings captured with an execution request."""

    max_attempts: int = Field(default=1, ge=1)
    base_delay_seconds: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    max_delay_seconds: float = Field(default=30.0, ge=0, allow_inf_nan=False)


class AccessContext(FrozenModel):
    """Trusted caller identity and workspace authorization scope."""

    principal_id: str = Field(min_length=1)
    authentication_kind: str = Field(min_length=1)
    roles: frozenset[str] = frozenset()
    authorized_workspace_ids: frozenset[str] = frozenset()
    authorize_local_workspaces: bool = False


class WorkspaceSelector(FrozenModel):
    """Reference to exactly one durable workspace identity."""

    workspace_id: str | None = Field(default=None, min_length=1, max_length=256)
    path: Path | None = None

    @model_validator(mode="after")
    def require_exactly_one_identity(self) -> "WorkspaceSelector":
        """Reject ambiguous selectors and selectors without an identity."""
        if (self.workspace_id is None) == (self.path is None):
            raise ValueError("exactly one of workspace_id or path is required")
        return self


class Workspace(FrozenModel):
    """Durable Nexus workspace identity after canonical path resolution."""

    workspace_id: str = Field(min_length=1, max_length=256)
    canonical_path: Path
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    config_reference: str | None = Field(default=None, min_length=1, max_length=2048)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def require_absolute_canonical_path(self) -> "Workspace":
        """Ensure storage never records a process-relative workspace path."""
        if not self.canonical_path.is_absolute():
            raise ValueError("canonical_path must be absolute")
        return self

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        """Store all domain timestamps in UTC."""
        return _normalize_utc(value)


class ProviderReference(FrozenModel):
    """A bounded provider-native identifier associated with a Nexus entity."""

    kind: str = Field(min_length=1, max_length=64)
    value: str = Field(min_length=1, max_length=4096)


class AgentSession(FrozenModel):
    """Durable conversation identity bound to one workspace and backend."""

    session_id: str = Field(min_length=1, max_length=256)
    workspace_id: str = Field(min_length=1, max_length=256)
    backend_id: str = Field(min_length=1, max_length=256)
    owner_id: str = Field(min_length=1, max_length=256)
    access_policy: AccessPolicy = "private"
    parent_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    provider_references: tuple[ProviderReference, ...] = ()
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at", "updated_at", mode="after")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        """Store all domain timestamps in UTC."""
        return _normalize_utc(value)


class JobAttempt(FrozenModel):
    """One persisted execution attempt in a job's retry history."""

    job_id: str = Field(min_length=1, max_length=256)
    attempt_number: int = Field(ge=1)
    phase: JobPhase = "claiming"
    worker_id: str | None = Field(default=None, min_length=1, max_length=256)
    lease_generation: int | None = Field(default=None, ge=1)
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    retry_classification: str | None = Field(default=None, min_length=1, max_length=256)
    reconciliation_classification: str | None = Field(default=None, min_length=1, max_length=256)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error_code: str | None = Field(default=None, min_length=1, max_length=128)
    error_message: str | None = Field(default=None, min_length=1, max_length=4096)

    @field_validator("lease_expires_at", "heartbeat_at", "started_at", "ended_at", mode="after")
    @classmethod
    def normalize_optional_timestamps(cls, value: datetime | None) -> datetime | None:
        """Store all domain timestamps in UTC when an attempt has reached that phase."""
        return None if value is None else _normalize_utc(value)


class BackendEvent(FrozenModel):
    """Normalized backend event before the store assigns job sequence information."""

    type: JobEventType
    payload: Mapping[str, JsonValue] = Field(default_factory=dict, validate_default=True)
    occurred_at: datetime = Field(default_factory=_utc_now)
    provider_event_type: str | None = Field(default=None, min_length=1, max_length=256)
    provider_reference: ProviderReference | None = None

    @property
    def event_type(self) -> JobEventType:
        """Return the normalized event type without shadowing Python's ``type`` at call sites."""
        return self.type

    @field_validator("payload", mode="after")
    @classmethod
    def freeze_payload(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Protect normalized content from mutation after event construction."""
        return _freeze_json_mapping(value)

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Emit ordinary JSON objects and arrays at the persistence boundary."""
        return _thaw_json_mapping(value)

    @field_validator("occurred_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Store event timestamps in UTC."""
        return _normalize_utc(value)


class JobEvent(FrozenModel):
    """Committed event with job-local order and an explicit payload schema revision."""

    job_id: str = Field(min_length=1, max_length=256)
    sequence: int = Field(ge=1)
    type: JobEventType
    payload: Mapping[str, JsonValue] = Field(default_factory=dict, validate_default=True)
    payload_schema_version: int = Field(default=1, ge=1)
    occurred_at: datetime = Field(default_factory=_utc_now)
    attempt_number: int | None = Field(default=None, ge=1)
    provider_event_type: str | None = Field(default=None, min_length=1, max_length=256)
    provider_reference: ProviderReference | None = None

    @property
    def event_type(self) -> JobEventType:
        """Return the normalized event type without shadowing Python's ``type`` at call sites."""
        return self.type

    @field_validator("payload", mode="after")
    @classmethod
    def freeze_payload(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Protect committed content from mutation after event construction."""
        return _freeze_json_mapping(value)

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Emit ordinary JSON objects and arrays at the persistence boundary."""
        return _thaw_json_mapping(value)

    @field_validator("occurred_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Store event timestamps in UTC."""
        return _normalize_utc(value)


class ExecutionConfigValues(FrozenModel):
    """A partial, credential-free execution-configuration layer."""

    model: str | None = Field(default=None, min_length=1, max_length=256)
    reasoning_effort: str | None = Field(default=None, min_length=1, max_length=64)
    sandbox: SandboxMode | None = None
    approval_policy: ApprovalPolicy | None = None
    timeout_seconds: int | None = Field(default=None, ge=1)
    output_limit_bytes: int | None = Field(default=None, ge=1)
    retry_policy: RetryPolicy | None = None


class ConfigLayerSnapshot(FrozenModel):
    """Immutable Nexus configuration values and the source from which they were captured."""

    values: ExecutionConfigValues
    source: str = Field(min_length=1, max_length=2048)
    source_hash: str = Field(min_length=64, max_length=64)
    captured_at: datetime = Field(default_factory=_utc_now)

    @field_validator("captured_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Store capture timestamps in UTC."""
        return _normalize_utc(value)


class RequestedExecutionConfig(FrozenModel):
    """Caller intent and immutable Nexus lower-precedence configuration snapshots."""

    explicit: ExecutionConfigValues = Field(default_factory=ExecutionConfigValues)
    workspace: ConfigLayerSnapshot | None = None
    user: ConfigLayerSnapshot | None = None
    environment: ConfigLayerSnapshot | None = None


class ResolvedExecutionConfig(FrozenModel):
    """Effective execution configuration frozen before provider side effects begin."""

    _FIELD_NAMES: ClassVar[tuple[ConfigFieldName, ...]] = (
        "model",
        "reasoning_effort",
        "sandbox",
        "approval_policy",
        "timeout_seconds",
        "output_limit_bytes",
        "retry_policy",
    )

    model: str | None = Field(default=None, min_length=1, max_length=256)
    reasoning_effort: str | None = Field(default=None, min_length=1, max_length=64)
    sandbox: SandboxMode | None = None
    approval_policy: ApprovalPolicy | None = None
    timeout_seconds: int | None = Field(default=None, ge=1)
    output_limit_bytes: int | None = Field(default=None, ge=1)
    retry_policy: RetryPolicy | None = None
    sources: Mapping[ConfigFieldName, ConfigSource] = Field(
        default_factory=dict, validate_default=True
    )

    @field_validator("sources", mode="after")
    @classmethod
    def freeze_sources(
        cls, value: Mapping[ConfigFieldName, ConfigSource]
    ) -> Mapping[ConfigFieldName, ConfigSource]:
        """Protect effective-value provenance from mutation after resolution."""
        return MappingProxyType(dict(value))

    @field_serializer("sources")
    def serialize_sources(
        self, value: Mapping[ConfigFieldName, ConfigSource]
    ) -> dict[ConfigFieldName, ConfigSource]:
        """Emit provenance as an ordinary dictionary at serialization boundaries."""
        return dict(value)

    @classmethod
    def from_requested(
        cls,
        requested: RequestedExecutionConfig,
        *,
        backend_defaults: ExecutionConfigValues | Mapping[str, object],
        fallback_defaults: ExecutionConfigValues | Mapping[str, object] | None = None,
        fallback_source: FallbackConfigSource = "fallback",
    ) -> "ResolvedExecutionConfig":
        """Resolve one immutable config using the approved precedence order.

        Provider defaults intentionally outrank stored Nexus layers. Mapping inputs support the
        empty mapping used at runtime by backends with no provider-native defaults while preserving
        the same credential-free validation as explicit model instances.
        """
        provider_values = _coerce_execution_values(backend_defaults)
        fallback_values = _coerce_execution_values(fallback_defaults)
        layers: list[tuple[ConfigSource, ExecutionConfigValues]] = [
            ("explicit", requested.explicit),
            ("provider", provider_values),
        ]
        if requested.workspace is not None:
            layers.append(("workspace", requested.workspace.values))
        if requested.user is not None:
            layers.append(("user", requested.user.values))
        if requested.environment is not None:
            layers.append(("environment", requested.environment.values))
        layers.append((fallback_source, fallback_values))
        values: dict[ConfigFieldName, ConfigValue | None] = dict.fromkeys(cls._FIELD_NAMES)
        sources: dict[ConfigFieldName, ConfigSource] = {}
        for source, layer in layers:
            for field_name in cls._FIELD_NAMES:
                value = cast("ConfigValue | None", getattr(layer, field_name))
                if values[field_name] is None and value is not None:
                    values[field_name] = value
                    sources[field_name] = source

        return cls(
            model=cast("str | None", values["model"]),
            reasoning_effort=cast("str | None", values["reasoning_effort"]),
            sandbox=cast("SandboxMode | None", values["sandbox"]),
            approval_policy=cast("ApprovalPolicy | None", values["approval_policy"]),
            timeout_seconds=cast("int | None", values["timeout_seconds"]),
            output_limit_bytes=cast("int | None", values["output_limit_bytes"]),
            retry_policy=cast("RetryPolicy | None", values["retry_policy"]),
            sources=sources,
        )


class AgentJob(FrozenModel):
    """One durable, typed operation and its immutable request snapshot."""

    job_id: str = Field(min_length=1, max_length=256)
    workspace_id: str = Field(min_length=1, max_length=256)
    backend_id: str = Field(min_length=1, max_length=256)
    owner_id: str = Field(min_length=1, max_length=256)
    operation: AgentOperation
    requested_config: RequestedExecutionConfig
    request_hash: str = Field(min_length=64, max_length=64)
    access_policy: AccessPolicy = "private"
    session_id: str | None = Field(default=None, min_length=1, max_length=256)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=512)
    source_checkpoint: tuple[ProviderReference, ...] = Field(default=(), max_length=256)
    state: JobState = "queued"
    resolved_config: ResolvedExecutionConfig | None = None
    cancel_requested_at: datetime | None = None
    lease_owner_id: str | None = Field(default=None, min_length=1, max_length=256)
    lease_generation: int | None = Field(default=None, ge=1)
    lease_expires_at: datetime | None = None
    retry_at: datetime | None = None
    terminal_result_reference: str | None = Field(default=None, min_length=1, max_length=4096)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None

    @property
    def operation_kind(self) -> OperationKind:
        """Return the operation discriminator through the closed public kind type."""
        return self.operation.kind

    @field_validator(
        "cancel_requested_at",
        "lease_expires_at",
        "retry_at",
        "created_at",
        "updated_at",
        "completed_at",
        mode="after",
    )
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        """Store all durable job timestamps in UTC."""
        return None if value is None else _normalize_utc(value)


class JobHandle(FrozenModel):
    """The stable identity returned when a typed operation is admitted."""

    job_id: str = Field(min_length=1, max_length=256)
    session_id: str | None = Field(default=None, min_length=1, max_length=256)
    operation: AgentOperation
    state: JobState = "queued"


class JobStatus(FrozenModel):
    """A bounded public projection of current durable job state."""

    job_id: str = Field(min_length=1, max_length=256)
    workspace_id: str = Field(min_length=1, max_length=256)
    backend_id: str = Field(min_length=1, max_length=256)
    session_id: str | None = Field(default=None, min_length=1, max_length=256)
    operation: AgentOperation
    state: JobState
    phase: JobPhase | None = None
    pending_inputs: tuple["PendingInput", ...] = Field(default=(), max_length=256)
    resolved_config: ResolvedExecutionConfig | None = None
    latest_event_sequence: int = Field(default=0, ge=0)
    cancel_requested: bool = False
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    @field_validator("created_at", "updated_at", "completed_at", mode="after")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        """Store public status timestamps in UTC."""
        return None if value is None else _normalize_utc(value)

    @model_validator(mode="after")
    def require_pending_inputs_to_be_unresolved(self) -> "JobStatus":
        """Keep resolved interaction records out of the authoritative pending collection."""
        if any(
            item.response is not None or item.resolved_at is not None
            for item in self.pending_inputs
        ):
            raise ValueError("pending_inputs must contain only unresolved inputs")
        return self

    @classmethod
    def from_job(
        cls,
        job: AgentJob,
        *,
        phase: JobPhase | None = None,
        pending_inputs: tuple["PendingInput", ...] = (),
        latest_event_sequence: int = 0,
    ) -> "JobStatus":
        """Project a durable job without exposing internal lease details."""
        return cls(
            job_id=job.job_id,
            workspace_id=job.workspace_id,
            backend_id=job.backend_id,
            session_id=job.session_id,
            operation=job.operation,
            state=job.state,
            phase=phase,
            pending_inputs=pending_inputs,
            resolved_config=job.resolved_config,
            latest_event_sequence=latest_event_sequence,
            cancel_requested=job.cancel_requested_at is not None,
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
        )


class JobListPage(FrozenModel):
    """An ordered page of public job status values and an opaque cursor."""

    items: tuple[JobStatus, ...] = Field(default=(), max_length=100)
    next_cursor: str | None = Field(default=None, min_length=1, max_length=4096)


def _coerce_execution_values(
    values: ExecutionConfigValues | Mapping[str, object] | None,
) -> ExecutionConfigValues:
    if values is None:
        return ExecutionConfigValues()
    if isinstance(values, ExecutionConfigValues):
        return values
    return ExecutionConfigValues.model_validate(values)


def new_id() -> str:
    """Return an opaque, randomly generated Nexus identifier."""
    return str(uuid.uuid4())


def validate_job_transition(current: JobState, target: JobState) -> None:
    """Raise when a requested public job-state transition is not legal."""
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidJobTransitionError(current=current, target=target)
