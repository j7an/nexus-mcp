"""Behavioral tests for the deterministic backend registry."""

import pytest

from nexus_mcp.backends import (
    BackendFailure,
    CompletedReconciliationOutcome,
)
from nexus_mcp.backends.manager import BackendManager
from nexus_mcp.core import (
    BackendAvailability,
    BackendEvent,
    JobAttempt,
    ProviderReference,
    QuestionRequest,
    QuestionResponse,
    ResolvedExecutionConfig,
    TurnOperation,
    UnsupportedCapabilityError,
)
from tests.fixtures import make_agent_job, make_job_error, make_turn_result, make_workspace
from tests.job_fakes import (
    EmitEventAction,
    RaiseFailureAction,
    RecordReferenceAction,
    RequestInputAction,
    ReturnReconciliationAction,
    ReturnResultAction,
    ScriptedBackend,
)


@pytest.fixture
def workspace():
    """Provide a stable workspace for backend health checks."""
    return make_workspace()


@pytest.fixture
def scripted_backend() -> ScriptedBackend:
    """Provide one available backend with all operations advertised."""
    return ScriptedBackend()


async def test_manager_rejects_unadvertised_operation(scripted_backend: ScriptedBackend):
    """Admission must reject an operation outside a backend's advertised capabilities."""
    manager = BackendManager([scripted_backend.with_operations({"turn"})])

    with pytest.raises(UnsupportedCapabilityError):
        manager.require_operation(scripted_backend.backend_id, "review")


async def test_manager_closes_each_runtime_once(scripted_backend: ScriptedBackend):
    """Runtime shutdown must not close a registered backend more than once."""
    manager = BackendManager([scripted_backend])

    await manager.close()
    await manager.close()

    assert scripted_backend.close_calls == 1


async def test_manager_lists_unavailable_backend_without_removing_it(
    scripted_backend: ScriptedBackend,
    workspace,
):
    """A transient health failure must leave backend identity and capabilities registered."""
    scripted_backend.availability = BackendAvailability(available=False, reason="not configured")

    statuses = await BackendManager([scripted_backend]).list_statuses(workspace)

    assert statuses[0].descriptor.backend_id == scripted_backend.backend_id
    assert statuses[0].availability.available is False


async def test_manager_lists_registered_backends_by_identifier(workspace):
    """Discovery results must remain deterministic regardless of registration order."""
    zeta = ScriptedBackend(backend_id="zeta")
    alpha = ScriptedBackend(backend_id="alpha")

    statuses = await BackendManager([zeta, alpha]).list_statuses(workspace)

    assert [status.descriptor.backend_id for status in statuses] == ["alpha", "zeta"]


def test_manager_rejects_duplicate_backend_identifiers():
    """One identifier must resolve to exactly one registered runtime."""
    duplicate = ScriptedBackend(backend_id="shared")

    with pytest.raises(ValueError, match="Duplicate backend id: shared"):
        BackendManager([duplicate, ScriptedBackend(backend_id="shared")])


class RecordingContext:
    """In-memory context that makes scripted backend effects observable."""

    def __init__(self) -> None:
        self.job = make_agent_job()
        self.attempt = JobAttempt(job_id=self.job.job_id, attempt_number=1)
        self.workspace = make_workspace()
        self.resolved_config = ResolvedExecutionConfig()
        self.events: list[BackendEvent] = []
        self.references: list[ProviderReference] = []
        self.input_requests: list[QuestionRequest] = []

    async def emit(self, event: BackendEvent) -> None:
        """Record the emitted event."""
        self.events.append(event)

    async def emit_output_delta(self, text: str) -> None:
        """Ignore output because this test exercises explicit event scripting."""

    async def record_provider_reference(self, reference: ProviderReference) -> None:
        """Record the persisted provider reference."""
        self.references.append(reference)

    async def request_input(self, request: QuestionRequest) -> QuestionResponse:
        """Record and answer the scripted question."""
        self.input_requests.append(request)
        return QuestionResponse(answer="continue")

    async def wait_for_control(self):
        """Control delivery is outside this scripted action sequence."""
        raise AssertionError("wait_for_control was not scripted")

    async def checkpoint(self) -> None:
        """No checkpoint is required for this action sequence."""


async def test_scripted_backend_replays_each_execution_action_once_in_order():
    """Worker tests can script backend effects without a real provider runtime."""
    backend = ScriptedBackend()
    context = RecordingContext()
    operation = TurnOperation(prompt="Inspect")
    event = BackendEvent(type="progress")
    reference = ProviderReference(kind="thread", value="thread-1")
    request = QuestionRequest(prompt="Continue?")
    result = make_turn_result()
    failure = BackendFailure(make_job_error(), "reconcile_required")
    backend.queue_execute(
        EmitEventAction(event),
        RecordReferenceAction(reference),
        RequestInputAction(request),
        RaiseFailureAction(failure),
    )

    with pytest.raises(BackendFailure) as raised:
        await backend.execute(operation, context)

    assert raised.value is failure
    assert context.events == [event]
    assert context.references == [reference]
    assert context.input_requests == [request]
    assert backend.execute_calls == [(operation, context)]

    backend.queue_execute(ReturnResultAction(result))

    assert await backend.execute(operation, context) is result
    assert backend.execute_calls == [(operation, context), (operation, context)]


async def test_scripted_backend_returns_and_records_reconciliation_outcomes():
    """No-replay tests can assert reconcile calls separately from execution calls."""
    backend = ScriptedBackend()
    context = RecordingContext()
    reference = ProviderReference(kind="thread", value="thread-1")
    outcome = CompletedReconciliationOutcome(result=make_turn_result())
    backend.queue_reconcile(ReturnReconciliationAction(outcome))

    assert await backend.reconcile((reference,), context) is outcome
    assert backend.reconcile_calls == [((reference,), context)]
