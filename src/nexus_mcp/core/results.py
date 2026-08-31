"""Normalized operation outcomes and public job-result responses."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
)

from nexus_mcp.core._json import (
    freeze_bounded_json_mapping,
    freeze_bounded_json_value,
    thaw_json_mapping,
    thaw_json_value,
)
from nexus_mcp.core.capabilities import BackendCapabilities
from nexus_mcp.core.models import AgentSession, JobState, ProviderReference
from nexus_mcp.core.operations import ReviewDelivery, ReviewTarget

__all__ = [
    "CancelReceipt",
    "CancelledJobResultResponse",
    "CommandSummary",
    "DiagnosticsResult",
    "ErrorRetryDisposition",
    "FailedJobResultResponse",
    "ForkResult",
    "JobError",
    "JobErrorCode",
    "JobResultEnvelope",
    "JobResultResponse",
    "OperationResult",
    "PendingJobResultResponse",
    "ReviewFinding",
    "ReviewResult",
    "ReviewSeverity",
    "SucceededJobResultResponse",
    "TurnResult",
]

type ReviewSeverity = Literal["critical", "high", "medium", "low", "info"]
type ErrorRetryDisposition = Literal["safe_to_retry", "reconcile_required", "terminal"]
type JobErrorCode = Literal[
    "backend_unknown",
    "backend_disabled",
    "unsupported_capability",
    "workspace_invalid",
    "session_not_found",
    "session_busy",
    "job_not_found",
    "input_not_found",
    "access_denied",
    "idempotency_conflict",
    "input_already_resolved",
    "stale_lease",
    "invalid_job_transition",
    "backend_unavailable",
    "authentication_required",
    "provider_failed",
    "timeout",
    "structured_output_invalid",
    "process_lost",
    "outcome_unknown",
    "internal_error",
]

_MAX_DETAILS_BYTES = 16_384


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be a timezone-aware UTC datetime")
    return value.astimezone(UTC)


class _ResultModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CommandSummary(_ResultModel):
    """A bounded semantic summary of one provider-observed command."""

    command: str = Field(min_length=1, max_length=4096)
    exit_code: int | None = None
    output_summary: str | None = Field(default=None, max_length=4096)


class ReviewFinding(_ResultModel):
    """One provider-neutral review finding."""

    severity: ReviewSeverity
    title: str = Field(min_length=1, max_length=512)
    evidence: str = Field(min_length=1, max_length=8192)
    source_location: str | None = Field(default=None, min_length=1, max_length=4096)
    recommended_action: str | None = Field(default=None, min_length=1, max_length=4096)


class TurnResult(_ResultModel):
    """Normalized successful turn output."""

    kind: Literal["turn"] = "turn"
    message: str = Field(max_length=1_048_576)
    structured_output: JsonValue | None = None
    changed_files: tuple[Annotated[str, Field(min_length=1, max_length=4096)], ...] = Field(
        default=(),
        max_length=4096,
    )
    commands: tuple[CommandSummary, ...] = Field(default=(), max_length=4096)
    usage: Mapping[str, JsonValue] = Field(
        default_factory=dict,
        max_length=256,
        validate_default=True,
    )

    @field_validator("structured_output", mode="after")
    @classmethod
    def freeze_structured_output(cls, value: JsonValue | None) -> JsonValue | None:
        """Bound and protect structured result data from post-validation mutation."""
        return None if value is None else freeze_bounded_json_value(value)

    @field_serializer("structured_output")
    def serialize_structured_output(self, value: JsonValue | None) -> JsonValue | None:
        """Emit ordinary JSON containers for structured output."""
        return None if value is None else thaw_json_value(value)

    @field_validator("usage", mode="after")
    @classmethod
    def freeze_usage(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Bound and protect normalized usage data from post-validation mutation."""
        return freeze_bounded_json_mapping(value)

    @field_serializer("usage")
    def serialize_usage(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Emit an ordinary JSON object for normalized usage."""
        return thaw_json_mapping(value)


class ReviewResult(_ResultModel):
    """A normalized review gate result with its admitted scope."""

    kind: Literal["review"] = "review"
    verdict: Literal["pass", "fail", "inconclusive"]
    summary: str = Field(min_length=1, max_length=131_072)
    target: ReviewTarget
    delivery: ReviewDelivery
    findings: tuple[ReviewFinding, ...] = Field(default=(), max_length=4096)


class DiagnosticsResult(_ResultModel):
    """Current normalized backend diagnostics."""

    kind: Literal["diagnostics"] = "diagnostics"
    available: bool
    authenticated: bool | None = None
    version: str | None = Field(default=None, min_length=1, max_length=256)
    setup_guidance: str | None = Field(default=None, min_length=1, max_length=4096)
    models: tuple[Annotated[str, Field(min_length=1, max_length=256)], ...] = Field(
        default=(),
        max_length=1024,
    )
    capabilities: BackendCapabilities = Field(
        default_factory=lambda: BackendCapabilities(operations=frozenset())
    )


class ForkResult(_ResultModel):
    """A child Nexus session and its provider lineage."""

    kind: Literal["fork"] = "fork"
    session: AgentSession
    provider_reference: ProviderReference | None = None
    parent_session_id: str = Field(min_length=1, max_length=256)


type OperationResult = Annotated[
    TurnResult | ReviewResult | DiagnosticsResult | ForkResult,
    Field(discriminator="kind"),
]


class JobError(_ResultModel):
    """A stable, user-safe job error with bounded provider diagnostics."""

    code: JobErrorCode
    message: str = Field(min_length=1, max_length=4096)
    retry_disposition: ErrorRetryDisposition = "terminal"
    recoverable: bool = False
    details: Mapping[str, JsonValue] = Field(
        default_factory=dict,
        max_length=256,
        validate_default=True,
    )

    @field_validator("details", mode="after")
    @classmethod
    def bound_and_freeze_details(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Enforce a canonical UTF-8 byte limit and freeze diagnostics recursively."""
        return freeze_bounded_json_mapping(value, max_bytes=_MAX_DETAILS_BYTES)

    @field_serializer("details")
    def serialize_details(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Emit an ordinary JSON object for durable diagnostics."""
        return thaw_json_mapping(value)


class JobResultEnvelope(_ResultModel):
    """One successful normalized operation payload bound to its job."""

    job_id: str = Field(min_length=1, max_length=256)
    payload: OperationResult
    completed_at: datetime = Field(default_factory=_utc_now)

    @field_validator("completed_at", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Store result completion timestamps in UTC."""
        return _normalize_utc(value)


class PendingJobResultResponse(_ResultModel):
    """A result-poll response for a nonterminal job."""

    status: Literal["pending"] = "pending"
    job_id: str = Field(min_length=1, max_length=256)
    state: Literal["queued", "running", "input_required"]


class SucceededJobResultResponse(_ResultModel):
    """A result-poll response for a completed job."""

    status: Literal["succeeded"] = "succeeded"
    job_id: str = Field(min_length=1, max_length=256)
    result: JobResultEnvelope


class FailedJobResultResponse(_ResultModel):
    """A result-poll response for a failed job."""

    status: Literal["failed"] = "failed"
    job_id: str = Field(min_length=1, max_length=256)
    error: JobError


class CancelledJobResultResponse(_ResultModel):
    """A result-poll response for a cancelled job."""

    status: Literal["cancelled"] = "cancelled"
    job_id: str = Field(min_length=1, max_length=256)


type JobResultResponse = Annotated[
    PendingJobResultResponse
    | SucceededJobResultResponse
    | FailedJobResultResponse
    | CancelledJobResultResponse,
    Field(discriminator="status"),
]


class CancelReceipt(_ResultModel):
    """The durable outcome of an idempotent cancellation request."""

    job_id: str = Field(min_length=1, max_length=256)
    state: JobState
    cancel_requested: bool
    completed_immediately: bool
    event_committed: bool
