"""Atomic domain-level persistence contracts for durable Nexus jobs."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nexus_mcp.core import (
    AccessPolicy,
    AgentJob,
    AgentOperation,
    AgentSession,
    BackendEvent,
    CancelReceipt,
    InputResolutionReceipt,
    InputResponse,
    JobAttempt,
    JobError,
    JobEvent,
    JobHandle,
    JobResultEnvelope,
    JobState,
    OperationResult,
    PendingInput,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    Workspace,
    WorkspaceSelector,
)

__all__ = [
    "CancelJobCommand",
    "CancelledTerminalOutcome",
    "ClaimedJob",
    "ControlSnapshot",
    "CreateJobCommand",
    "CreateJobResult",
    "EventPage",
    "FailedTerminalOutcome",
    "JobAccessFilter",
    "JobQuery",
    "JobStore",
    "LeaseToken",
    "PrunePolicy",
    "PruneResult",
    "ResolveInputCommand",
    "RuntimeLease",
    "RuntimeLeaseBusyError",
    "StoredJobPage",
    "SucceededTerminalOutcome",
    "TerminalOutcome",
]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be a timezone-aware UTC datetime")
    return value.astimezone(UTC)


class _StoreModel(BaseModel):
    """Shared immutable and closed validation for persisted store commands."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CreateJobCommand(_StoreModel):
    """One fully admitted request whose initial snapshot and event commit atomically."""

    workspace: Workspace
    backend_id: str = Field(min_length=1, max_length=256)
    owner_id: str = Field(min_length=1, max_length=256)
    access_policy: AccessPolicy
    operation: AgentOperation
    requested_config: RequestedExecutionConfig
    session_id: str | None = Field(default=None, min_length=1, max_length=256)
    create_session: bool = False
    parent_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    source_checkpoint: tuple[ProviderReference, ...] = Field(default=(), max_length=256)
    command_family: str = Field(min_length=1, max_length=128)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=512)
    queued_event: BackendEvent

    @property
    def source_session_id(self) -> str | None:
        """Return the session from which continuation or child work derives."""
        if self.parent_session_id is not None:
            return self.parent_session_id
        if not self.create_session:
            return self.session_id
        return None

    @model_validator(mode="after")
    def validate_session_shape(self) -> "CreateJobCommand":
        """Keep diagnostics sessionless and every conversational operation session-bound."""
        if self.create_session and self.session_id is None:
            raise ValueError("create_session=True requires session_id")
        if self.operation.kind == "diagnostics":
            if self.session_id is not None or self.create_session:
                raise ValueError("diagnostics operations must be sessionless")
        elif self.session_id is None:
            raise ValueError("non-diagnostics operations require session_id")
        if self.source_checkpoint and self.source_session_id is None:
            raise ValueError("source_checkpoint requires a source session")
        return self


def _create_job_request_hash(command: CreateJobCommand) -> str:
    """Hash caller intent without generated identities or recaptured persistence state."""
    payload = {
        "version": 1,
        "owner_id": command.owner_id,
        "workspace_id": command.workspace.workspace_id,
        "backend_id": command.backend_id,
        "access_policy": command.access_policy,
        "operation": command.operation.model_dump(mode="json"),
        "explicit_config": command.requested_config.explicit.model_dump(mode="json"),
        "source_session_id": command.source_session_id,
        "command_family": command.command_family,
        "create_session": command.create_session,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CreateJobResult(_StoreModel):
    """The stable admission handle and whether this call created it."""

    handle: JobHandle
    created: bool


class LeaseToken(_StoreModel):
    """Generation-fenced authority for one exact persisted job attempt."""

    job_id: str = Field(min_length=1, max_length=256)
    owner_id: str = Field(min_length=1, max_length=256)
    generation: int = Field(ge=1)
    attempt_number: int = Field(ge=1)


class ClaimedJob(_StoreModel):
    """The immutable snapshots returned by one atomic worker claim."""

    job: AgentJob
    attempt: JobAttempt
    token: LeaseToken


class JobAccessFilter(_StoreModel):
    """Trusted principal visibility applied to a stored-job query."""

    principal_id: str = Field(min_length=1, max_length=256)
    workspace_authorized: bool = False


class JobQuery(_StoreModel):
    """A bounded, authorized query over stored job snapshots."""

    workspace_id: str = Field(min_length=1, max_length=256)
    access: JobAccessFilter
    states: frozenset[JobState] = Field(min_length=1)
    limit: int = Field(ge=1, le=100)
    cursor: str | None = Field(default=None, min_length=1, max_length=4096)


class StoredJobPage(_StoreModel):
    """One deterministic descending page of durable job snapshots."""

    jobs: tuple[AgentJob, ...] = Field(default=(), max_length=100)
    next_cursor: str | None = Field(default=None, min_length=1, max_length=4096)


class EventPage(_StoreModel):
    """Committed job events strictly after one job-local sequence cursor."""

    events: tuple[JobEvent, ...] = Field(default=(), max_length=1000)
    next_after_sequence: int | None = Field(default=None, ge=1)
    has_more: bool = False


class ControlSnapshot(_StoreModel):
    """Current job controls visible only to the matching worker fence."""

    state: JobState
    cancel_requested: bool
    unresolved_inputs: tuple[PendingInput, ...] = Field(default=(), max_length=256)
    resolved_inputs: tuple[PendingInput, ...] = Field(default=(), max_length=256)
    lease_generation: int = Field(ge=1)

    @model_validator(mode="after")
    def require_truthful_input_partitions(self) -> "ControlSnapshot":
        """Keep pending requests and committed responses in disjoint typed partitions."""
        if any(item.response is not None for item in self.unresolved_inputs):
            raise ValueError("unresolved_inputs must not contain responses")
        if any(item.response is None for item in self.resolved_inputs):
            raise ValueError("resolved_inputs must contain responses")
        unresolved_ids = {item.input_id for item in self.unresolved_inputs}
        resolved_ids = {item.input_id for item in self.resolved_inputs}
        if unresolved_ids & resolved_ids:
            raise ValueError("input partitions must be disjoint")
        return self


class ResolveInputCommand(_StoreModel):
    """One externally supplied response and its atomic semantic event."""

    job_id: str = Field(min_length=1, max_length=256)
    input_id: str = Field(min_length=1, max_length=256)
    response: InputResponse
    resolved_at: datetime = Field(default_factory=_utc_now)
    event: BackendEvent

    @field_validator("resolved_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Persist the resolution timestamp in UTC."""
        return _normalize_utc(value)


class CancelJobCommand(_StoreModel):
    """One atomic cancellation decision with state-specific semantic events."""

    job_id: str = Field(min_length=1, max_length=256)
    active_cancellation_allowed: bool
    requested_at: datetime = Field(default_factory=_utc_now)
    queued_event: BackendEvent
    active_event: BackendEvent

    @field_validator("requested_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Persist the cancellation timestamp in UTC."""
        return _normalize_utc(value)

    @model_validator(mode="after")
    def require_truthful_state_events(self) -> "CancelJobCommand":
        """Prevent queued completion and active intent from using misleading event types."""
        if self.queued_event.type != "job_cancelled":
            raise ValueError("queued_event must be job_cancelled")
        if self.active_event.type != "cancel_requested":
            raise ValueError("active_event must be cancel_requested")
        return self


class SucceededTerminalOutcome(_StoreModel):
    """A successful normalized operation result."""

    kind: Literal["succeeded"] = "succeeded"
    result: OperationResult
    completed_at: datetime = Field(default_factory=_utc_now)

    @field_validator("completed_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Persist the completion timestamp in UTC."""
        return _normalize_utc(value)


class FailedTerminalOutcome(_StoreModel):
    """A normalized terminal job error."""

    kind: Literal["failed"] = "failed"
    error: JobError
    completed_at: datetime = Field(default_factory=_utc_now)

    @field_validator("completed_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Persist the failure timestamp in UTC."""
        return _normalize_utc(value)


class CancelledTerminalOutcome(_StoreModel):
    """A confirmed terminal cancellation."""

    kind: Literal["cancelled"] = "cancelled"
    completed_at: datetime = Field(default_factory=_utc_now)

    @field_validator("completed_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Persist the cancellation timestamp in UTC."""
        return _normalize_utc(value)


type TerminalOutcome = Annotated[
    SucceededTerminalOutcome | FailedTerminalOutcome | CancelledTerminalOutcome,
    Field(discriminator="kind"),
]


class RuntimeLease(_StoreModel):
    """Generation-fenced ownership of one shared managed backend runtime."""

    runtime_key: str = Field(min_length=1, max_length=512)
    owner_id: str = Field(min_length=1, max_length=256)
    generation: int = Field(ge=1)
    lease_until: datetime
    heartbeat_at: datetime
    endpoint: str | None = Field(default=None, min_length=1, max_length=4096)

    @field_validator("lease_until", "heartbeat_at", mode="after")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        """Persist runtime coordination timestamps in UTC."""
        return _normalize_utc(value)


class RuntimeLeaseBusyError(Exception):
    """Raised when a different process owns an unexpired runtime lease."""

    def __init__(self, runtime_key: str, owner_id: str, lease_until: datetime) -> None:
        self.runtime_key = runtime_key
        self.owner_id = owner_id
        self.lease_until = lease_until
        super().__init__(
            f"Runtime {runtime_key} is leased by {owner_id} until {lease_until.isoformat()}"
        )


class PrunePolicy(_StoreModel):
    """Independent retention cutoffs for terminal jobs, events, and raw diagnostics."""

    terminal_job_before: datetime | None = None
    event_before: datetime | None = None
    raw_diagnostic_before: datetime | None = None
    raw_diagnostic_max_bytes: int | None = Field(default=None, gt=0)

    @field_validator(
        "terminal_job_before",
        "event_before",
        "raw_diagnostic_before",
        mode="after",
    )
    @classmethod
    def normalize_cutoffs(cls, value: datetime | None) -> datetime | None:
        """Persist retention cutoffs in UTC when supplied."""
        return None if value is None else _normalize_utc(value)

    @model_validator(mode="after")
    def require_complete_raw_diagnostic_policy(self) -> "PrunePolicy":
        """Require age and byte constraints together for raw diagnostic retention."""
        has_cutoff = self.raw_diagnostic_before is not None
        has_cap = self.raw_diagnostic_max_bytes is not None
        if has_cutoff != has_cap:
            raise ValueError(
                "raw_diagnostic_before and raw_diagnostic_max_bytes must be supplied together"
            )
        return self


class PruneResult(_StoreModel):
    """Counts deleted by one atomic retention pass."""

    terminal_jobs_deleted: int = Field(default=0, ge=0)
    events_deleted: int = Field(default=0, ge=0)
    raw_diagnostics_deleted: int = Field(default=0, ge=0)


class JobStore(Protocol):
    """Atomic domain operations implemented by each durable store."""

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def resolve_workspace(self, selector: WorkspaceSelector) -> Workspace: ...

    async def create_job(self, command: CreateJobCommand) -> CreateJobResult: ...

    async def get_session(self, session_id: str) -> AgentSession | None: ...

    async def get_job(self, job_id: str) -> AgentJob | None: ...

    async def get_job_result(self, job_id: str) -> JobResultEnvelope | JobError | None: ...

    async def get_job_attempts(self, job_id: str) -> tuple[JobAttempt, ...]: ...

    async def get_provider_references(
        self, *, session_id: str | None = None, job_id: str | None = None
    ) -> tuple[ProviderReference, ...]: ...

    async def get_pending_inputs(self, job_id: str) -> tuple[PendingInput, ...]: ...

    async def list_jobs(self, query: JobQuery) -> StoredJobPage: ...

    async def claim_next(
        self,
        owner_id: str,
        lease_until: datetime,
        *,
        event: BackendEvent,
    ) -> ClaimedJob | None: ...

    async def renew_lease(self, token: LeaseToken, lease_until: datetime) -> bool: ...

    async def store_resolved_config(
        self, token: LeaseToken, config: ResolvedExecutionConfig
    ) -> None: ...

    async def record_provider_reference(
        self, token: LeaseToken, reference: ProviderReference
    ) -> None: ...

    async def append_events(
        self, token: LeaseToken, events: tuple[BackendEvent, ...]
    ) -> tuple[JobEvent, ...]: ...

    async def mark_input_required(
        self,
        token: LeaseToken,
        inputs: tuple[PendingInput, ...],
        *,
        event: BackendEvent,
    ) -> None: ...

    async def mark_running(
        self,
        token: LeaseToken,
        resolved_input_ids: tuple[str, ...],
        *,
        event: BackendEvent,
    ) -> None: ...

    async def mark_reconciling(
        self,
        token: LeaseToken,
        error: JobError,
        *,
        event: BackendEvent,
    ) -> None: ...

    async def schedule_retry(
        self,
        token: LeaseToken,
        retry_at: datetime,
        error: JobError,
        *,
        event: BackendEvent,
    ) -> None: ...

    async def get_control_snapshot(self, token: LeaseToken) -> ControlSnapshot: ...

    async def resolve_input(self, command: ResolveInputCommand) -> InputResolutionReceipt: ...

    async def request_cancel(self, command: CancelJobCommand) -> CancelReceipt: ...

    async def terminalize(
        self,
        token: LeaseToken,
        outcome: TerminalOutcome,
        *,
        event: BackendEvent,
    ) -> AgentJob: ...

    async def read_events(self, job_id: str, after_sequence: int, limit: int) -> EventPage: ...

    async def acquire_runtime_lease(
        self, runtime_key: str, owner_id: str, lease_until: datetime
    ) -> RuntimeLease: ...

    async def renew_runtime_lease(self, lease: RuntimeLease, lease_until: datetime) -> bool: ...

    async def release_runtime_lease(self, lease: RuntimeLease) -> None: ...

    async def prune(self, policy: PrunePolicy, now: datetime) -> PruneResult: ...
