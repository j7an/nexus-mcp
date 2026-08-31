"""Framework-independent backend execution contracts."""

from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from nexus_mcp.core import (
    AgentJob,
    AgentOperation,
    BackendAvailability,
    BackendDescriptor,
    BackendEvent,
    InputRequest,
    InputResponse,
    JobAttempt,
    JobError,
    OperationResult,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    Workspace,
)

__all__ = [
    "ActiveReconciliationOutcome",
    "AgentBackend",
    "BackendExecutionContext",
    "BackendFailure",
    "CancelRequested",
    "CancelledReconciliationOutcome",
    "CompletedReconciliationOutcome",
    "ControlSignal",
    "FailedReconciliationOutcome",
    "InputRequiredReconciliationOutcome",
    "InputResolved",
    "LeaseLost",
    "ReconciliationOutcome",
    "RetryDisposition",
    "RuntimeShutdown",
    "UnknownReconciliationOutcome",
]

type RetryDisposition = Literal["safe_to_retry", "reconcile_required", "terminal"]


class _Signal(BaseModel):
    """Shared immutable validation for runtime control signals."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CancelRequested(_Signal):
    """Signal that a worker must stop execution at the next safe boundary."""

    kind: Literal["cancel_requested"] = "cancel_requested"


class InputResolved(_Signal):
    """Signal that a particular persisted input request now has a response."""

    kind: Literal["input_resolved"] = "input_resolved"
    input_id: str = Field(min_length=1, max_length=256)


class LeaseLost(_Signal):
    """Signal that this worker may no longer mutate the current job attempt."""

    kind: Literal["lease_lost"] = "lease_lost"


class RuntimeShutdown(_Signal):
    """Signal that the runtime is draining and execution must relinquish its lease."""

    kind: Literal["runtime_shutdown"] = "runtime_shutdown"


type ControlSignal = CancelRequested | InputResolved | LeaseLost | RuntimeShutdown


class _ReconciliationOutcome(BaseModel):
    """Shared immutable validation for reconciliation decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ActiveReconciliationOutcome(_ReconciliationOutcome):
    """The provider is still executing the previously started operation."""

    kind: Literal["active"] = "active"


class InputRequiredReconciliationOutcome(_ReconciliationOutcome):
    """The provider is waiting on an already persisted normalized input request."""

    kind: Literal["input_required"] = "input_required"


class CompletedReconciliationOutcome(_ReconciliationOutcome):
    """The provider completed with one normalized operation result."""

    kind: Literal["completed"] = "completed"
    result: OperationResult


class FailedReconciliationOutcome(_ReconciliationOutcome):
    """The provider reached a normalized terminal error."""

    kind: Literal["failed"] = "failed"
    error: JobError


class CancelledReconciliationOutcome(_ReconciliationOutcome):
    """The provider confirms the operation was cancelled."""

    kind: Literal["cancelled"] = "cancelled"


class UnknownReconciliationOutcome(_ReconciliationOutcome):
    """The provider state cannot be determined safely after an interrupted attempt."""

    kind: Literal["unknown"] = "unknown"
    error: JobError


type ReconciliationOutcome = Annotated[
    ActiveReconciliationOutcome
    | InputRequiredReconciliationOutcome
    | CompletedReconciliationOutcome
    | FailedReconciliationOutcome
    | CancelledReconciliationOutcome
    | UnknownReconciliationOutcome,
    Field(discriminator="kind"),
]


class BackendFailure(Exception):
    """A backend-raised, normalized failure classified for worker retry handling."""

    def __init__(self, error: JobError, retry_disposition: RetryDisposition) -> None:
        self.error = error
        self.retry_disposition = retry_disposition
        super().__init__(error.message)


class BackendExecutionContext(Protocol):
    """Worker-owned execution effects available to a backend without store access."""

    job: AgentJob
    attempt: JobAttempt
    workspace: Workspace
    resolved_config: ResolvedExecutionConfig

    async def emit(self, event: BackendEvent) -> None:
        """Persist one normalized backend event."""

    async def emit_output_delta(self, text: str) -> None:
        """Persist a normalized incremental output update."""

    async def record_provider_reference(self, reference: ProviderReference) -> None:
        """Persist one provider-native reference associated with the job."""

    async def request_input(self, request: InputRequest) -> InputResponse:
        """Persist a normalized input request and wait for its validated response."""

    async def wait_for_control(self) -> ControlSignal:
        """Wait for one worker control signal."""

    async def checkpoint(self) -> None:
        """Verify that the worker still owns the execution lease."""


class AgentBackend(Protocol):
    """Provider runtime boundary with no transport-framework or job-store dependency."""

    descriptor: BackendDescriptor

    async def check_availability(self, workspace: Workspace) -> BackendAvailability:
        """Return current health without altering registration or capabilities."""

    async def resolve_execution_config(
        self, requested: RequestedExecutionConfig, workspace: Workspace
    ) -> ResolvedExecutionConfig:
        """Resolve one credential-free configuration before provider side effects."""

    async def execute(
        self, operation: AgentOperation, context: BackendExecutionContext
    ) -> OperationResult:
        """Execute exactly one admitted operation."""

    async def reconcile(
        self,
        provider_state: tuple[ProviderReference, ...],
        context: BackendExecutionContext,
    ) -> ReconciliationOutcome:
        """Classify provider state after an interrupted execution attempt."""

    async def close(self) -> None:
        """Release provider runtime resources exactly once per manager lifecycle."""
