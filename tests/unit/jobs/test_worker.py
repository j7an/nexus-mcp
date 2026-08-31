"""Durable execution, retry, and reconciliation contracts for job workers."""

import asyncio
from pathlib import Path

import pytest

from nexus_mcp.backends import (
    ActiveReconciliationOutcome,
    BackendExecutionContext,
    BackendFailure,
    CompletedReconciliationOutcome,
    UnknownReconciliationOutcome,
)
from nexus_mcp.backends.manager import BackendManager
from nexus_mcp.core import (
    BackendAvailability,
    BackendEvent,
    DiagnosticsResult,
    ExecutionConfigValues,
    JobError,
    ProviderReference,
    RequestedExecutionConfig,
    RetryPolicy,
    TurnOperation,
    Workspace,
    WorkspaceInvalidError,
    WorkspaceSelector,
)
from nexus_mcp.exceptions import ConfigurationError
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.store import CreateJobCommand
from nexus_mcp.jobs.worker import JobWorker, WorkerPolicy, WorkerPool
from tests.fixtures import make_turn_result
from tests.job_fakes import (
    EmitEventAction,
    InMemoryJobStore,
    RaiseFailureAction,
    RecordReferenceAction,
    ReturnReconciliationAction,
    ReturnResultAction,
    ScriptedBackend,
)


def make_command(
    *,
    backend_id: str = "scripted",
    session_id: str = "session-worker",
    requested_config: RequestedExecutionConfig | None = None,
) -> CreateJobCommand:
    """Build one admitted turn using a real durable store command."""
    return CreateJobCommand(
        workspace=Workspace(
            workspace_id="workspace-worker",
            canonical_path=Path(__file__).resolve().parents[3],
        ),
        backend_id=backend_id,
        owner_id="local:worker-test",
        access_policy="private",
        operation=TurnOperation(prompt="Execute the durable task"),
        requested_config=requested_config or RequestedExecutionConfig(),
        session_id=session_id,
        create_session=True,
        command_family="worker-test",
        queued_event=BackendEvent(type="job_queued", payload={"state": "queued"}),
    )


async def admit(store: InMemoryJobStore, **overrides: object):
    """Persist and return one queued job."""
    command = make_command(**overrides)
    created = await store.create_job(command)
    job = await store.get_job(created.handle.job_id)
    assert job is not None
    return job


def make_worker(
    store: InMemoryJobStore,
    backend: ScriptedBackend,
    notifier: EventNotifier | None = None,
    **overrides: object,
) -> JobWorker:
    """Construct one worker with deterministic no-wait retry scheduling."""
    return JobWorker(
        worker_id="worker-1",
        store=store,
        backends=BackendManager([backend]),
        notifier=notifier or EventNotifier(),
        retry_delay=lambda attempt, retry_after, policy: 0.0,
        **overrides,
    )


async def test_worker_freezes_configuration_before_execute():
    """Provider execution cannot begin before its effective config is durable."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(
        store,
        requested_config=RequestedExecutionConfig(
            explicit=ExecutionConfigValues(model="provider/model")
        ),
    )
    trace: list[str] = []
    store.trace = backend.trace = trace
    backend.queue_execute(ReturnResultAction(make_turn_result()))

    assert await make_worker(store, backend).run_once() is True

    persisted = await store.get_job(job.job_id)
    assert persisted is not None
    assert persisted.resolved_config is not None
    assert persisted.resolved_config.model == "provider/model"
    assert trace.index("store_resolved_config") < trace.index("backend.execute")
    context = backend.execute_calls[0][1]
    assert context.attempt.phase == "executing"
    assert context.job.state == "running"
    assert context.job.resolved_config == context.resolved_config


async def test_worker_terminalizes_result_and_normalized_events_with_current_fence():
    """Semantic event fields survive in order while opaque provider payloads are discarded."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    reference = ProviderReference(kind="thread", value="thread-safe")
    backend.queue_execute(
        EmitEventAction(
            BackendEvent(
                type="provider_connected",
                payload={"backend_id": "scripted", "raw_payload": {"secret": "drop"}},
                provider_event_type="session.started",
                provider_reference=reference,
            )
        ),
        EmitEventAction(
            BackendEvent(
                type="command",
                payload={"command": "pwd", "exit_code": 0, "raw_payload": "drop"},
            )
        ),
        EmitEventAction(
            BackendEvent(
                type="file_change",
                payload={"path": "README.md", "status": "modified", "raw": "drop"},
            )
        ),
        EmitEventAction(
            BackendEvent(
                type="usage",
                payload={"input_tokens": 7, "output_tokens": 3, "opaque": {"drop": True}},
            )
        ),
        ReturnResultAction(make_turn_result(message="Complete final message")),
    )

    await make_worker(store, backend).run_once()

    result = await store.get_job_result(job.job_id)
    assert result is not None
    assert result.payload.kind == "turn"  # type: ignore[union-attr]
    events = (await store.read_events(job.job_id, 0, 100)).events
    semantic = [
        event
        for event in events
        if event.type in {"provider_connected", "command", "file_change", "usage", "message"}
    ]
    assert [event.type for event in semantic] == [
        "provider_connected",
        "command",
        "file_change",
        "usage",
        "message",
    ]
    assert dict(semantic[0].payload) == {"backend_id": "scripted"}
    assert dict(semantic[1].payload) == {"command": "pwd", "exit_code": 0}
    assert dict(semantic[2].payload) == {"path": "README.md", "status": "modified"}
    assert dict(semantic[3].payload) == {"input_tokens": 7, "output_tokens": 3}
    assert dict(semantic[4].payload) == {"text": "Complete final message", "final": True}
    assert semantic[0].provider_reference == reference
    assert "drop" not in str([event.model_dump(mode="json") for event in semantic])


async def test_worker_persists_the_complete_final_turn_message():
    """The authoritative final message is not truncated to an incremental chunk size."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    complete = "final-" * 1000
    backend.queue_execute(ReturnResultAction(make_turn_result(message=complete)))

    await make_worker(store, backend, output_chunk_bytes=32).run_once()

    messages = [
        event
        for event in (await store.read_events(job.job_id, 0, 100)).events
        if event.type == "message"
    ]
    assert dict(messages[-1].payload) == {"text": complete, "final": True}


async def test_reasoning_events_persist_metadata_without_reasoning_content():
    """Provider reasoning streams retain bounded status metadata but not opaque chain content."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    backend.queue_execute(
        EmitEventAction(
            BackendEvent(
                type="progress",
                provider_event_type="reasoning.delta",
                payload={
                    "stage": "analysis",
                    "status": "active",
                    "message": "private provider reasoning",
                    "raw_payload": {"secret": True},
                },
            )
        ),
        ReturnResultAction(make_turn_result()),
    )

    await make_worker(store, backend).run_once()

    reasoning = next(
        event
        for event in (await store.read_events(job.job_id, 0, 100)).events
        if event.provider_event_type == "reasoning.delta"
    )
    assert dict(reasoning.payload) == {"stage": "analysis", "status": "active"}


async def test_mismatched_backend_result_becomes_internal_error_without_escaping():
    """An invalid provider result kind cannot leave the claimed job leased forever."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    backend.queue_execute(ReturnResultAction(DiagnosticsResult(available=True)))

    await make_worker(store, backend).run_once()

    result = await store.get_job_result(job.job_id)
    assert isinstance(result, JobError)
    assert result.code == "internal_error"


@pytest.mark.parametrize(
    ("availability", "error_code"),
    [
        (BackendAvailability(available=False, reason="binary missing"), "backend_unavailable"),
        (
            BackendAvailability(available=True, authenticated=False, reason="login required"),
            "authentication_required",
        ),
    ],
)
async def test_backend_health_failure_becomes_durable_without_unregistering(
    availability: BackendAvailability,
    error_code: str,
):
    """Transient availability and auth observations fail only the claimed job."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    backend.availability = availability
    job = await admit(store)
    manager = BackendManager([backend])
    worker = JobWorker(
        worker_id="worker-health",
        store=store,
        backends=manager,
        notifier=EventNotifier(),
    )

    await worker.run_once()

    failed = await store.get_job_result(job.job_id)
    assert isinstance(failed, JobError)
    assert failed.code == error_code
    assert manager.get("scripted") is backend
    assert backend.execute_calls == []


async def test_backend_lookup_error_becomes_typed_durable_failure():
    """A stale admitted backend id cannot escape run_once or leave the job leased."""
    store = InMemoryJobStore()
    backend = ScriptedBackend(backend_id="registered")
    job = await admit(store, backend_id="missing")

    await make_worker(store, backend).run_once()

    result = await store.get_job_result(job.job_id)
    assert isinstance(result, JobError)
    assert result.code == "backend_unknown"
    assert backend.execute_calls == []


class WorkspaceErrorStore(InMemoryJobStore):
    """Store that reports a durable workspace resolution failure after claim."""

    async def resolve_workspace(self, selector: WorkspaceSelector) -> Workspace:
        raise WorkspaceInvalidError(selector.workspace_id or "unknown", "workspace disappeared")


async def test_workspace_resolution_error_becomes_typed_durable_failure():
    """A post-admission workspace failure terminalizes under the current fence."""
    store = WorkspaceErrorStore()
    backend = ScriptedBackend()
    job = await admit(store)

    await make_worker(store, backend).run_once()

    result = await store.get_job_result(job.job_id)
    assert isinstance(result, JobError)
    assert result.code == "workspace_invalid"
    assert backend.execute_calls == []


class AvailabilityErrorBackend(ScriptedBackend):
    """Backend whose health observation raises before any provider execution side effect."""

    async def check_availability(self, workspace: Workspace) -> BackendAvailability:
        raise RuntimeError("secret availability diagnostic")


async def test_availability_exception_becomes_sanitized_internal_error():
    """Unexpected health-probe failure is durable and does not disclose raw diagnostics."""
    store = InMemoryJobStore()
    backend = AvailabilityErrorBackend()
    job = await admit(store)

    await make_worker(store, backend).run_once()

    result = await store.get_job_result(job.job_id)
    assert isinstance(result, JobError)
    assert result.code == "internal_error"
    assert "secret" not in result.message
    assert backend.config_calls == []
    assert backend.execute_calls == []


class ConfigurationErrorOnceBackend(ScriptedBackend):
    """Backend whose first config resolution fails before later jobs can execute."""

    def __init__(self) -> None:
        super().__init__()
        self.resolve_attempts = 0

    async def resolve_execution_config(
        self,
        requested: RequestedExecutionConfig,
        workspace: Workspace,
    ):
        self.resolve_attempts += 1
        if self.resolve_attempts == 1:
            raise ConfigurationError("secret invalid provider configuration")
        return await super().resolve_execution_config(requested, workspace)


async def test_config_exception_terminalizes_and_pool_continues_to_next_job():
    """A pre-provider config failure cannot terminate the worker loop or block later work."""
    store = InMemoryJobStore()
    notifier = EventNotifier()
    backend = ConfigurationErrorOnceBackend()
    first = await admit(store, session_id="session-config-first")
    second = await admit(store, session_id="session-config-second")
    backend.queue_execute(ReturnResultAction(make_turn_result(message="second job completed")))
    pool = WorkerPool(
        store=store,
        backends=BackendManager([backend]),
        notifier=notifier,
        worker_count=1,
        retry_delay=lambda attempt, retry_after, policy: 0.0,
    )

    assert await pool.run_until_idle() == 2

    results = {
        first.job_id: await store.get_job_result(first.job_id),
        second.job_id: await store.get_job_result(second.job_id),
    }
    errors = [result for result in results.values() if isinstance(result, JobError)]
    successes = [result for result in results.values() if not isinstance(result, JobError)]
    assert len(errors) == 1 and errors[0].code == "internal_error"
    assert "secret" not in errors[0].message
    assert len(successes) == 1 and successes[0] is not None
    assert len(backend.execute_calls) == 1


async def test_reconcile_required_imports_completion_without_replaying_execute():
    """Once a provider reference exists, only reconciliation may finish the operation."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    reference = ProviderReference(kind="thread", value="thread-reconcile")
    uncertain = JobError(
        code="process_lost",
        message="Provider observation disconnected",
        retry_disposition="reconcile_required",
        recoverable=True,
    )
    backend.queue_execute(
        RecordReferenceAction(reference),
        RaiseFailureAction(BackendFailure(uncertain, "reconcile_required")),
    )
    backend.queue_reconcile(
        ReturnReconciliationAction(
            CompletedReconciliationOutcome(result=make_turn_result(message="Imported"))
        )
    )

    await make_worker(store, backend).run_once()

    result = await store.get_job_result(job.job_id)
    assert result is not None
    assert result.payload.message == "Imported"  # type: ignore[union-attr]
    assert len(backend.execute_calls) == 1
    assert len(backend.reconcile_calls) == 1
    assert backend.reconcile_calls[0][0] == (reference,)


async def test_unknown_reconciliation_terminalizes_outcome_unknown_without_replay():
    """An unknowable provider outcome is terminal rather than an implicit second execution."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    reference = ProviderReference(kind="thread", value="thread-unknown")
    failure = JobError(
        code="process_lost",
        message="Lost observation",
        retry_disposition="reconcile_required",
    )
    unknown = JobError(code="outcome_unknown", message="Provider outcome is unknown")
    backend.queue_execute(
        RecordReferenceAction(reference),
        RaiseFailureAction(BackendFailure(failure, "reconcile_required")),
    )
    backend.queue_reconcile(ReturnReconciliationAction(UnknownReconciliationOutcome(error=unknown)))

    await make_worker(store, backend).run_once()

    assert await store.get_job_result(job.job_id) == unknown
    assert len(backend.execute_calls) == 1
    assert len(backend.reconcile_calls) == 1


async def test_mismatched_reconciliation_result_terminalizes_internal_error_without_replay():
    """Imported completion must match the admitted operation before output or terminal commit."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    reference = ProviderReference(kind="thread", value="thread-mismatch")
    uncertain = JobError(
        code="process_lost",
        message="Provider observation disconnected",
        retry_disposition="reconcile_required",
    )
    backend.queue_execute(
        RecordReferenceAction(reference),
        RaiseFailureAction(BackendFailure(uncertain, "reconcile_required")),
    )
    backend.queue_reconcile(
        ReturnReconciliationAction(
            CompletedReconciliationOutcome(result=DiagnosticsResult(available=True))
        )
    )

    await make_worker(store, backend).run_once()

    result = await store.get_job_result(job.job_id)
    assert isinstance(result, JobError)
    assert result.code == "internal_error"
    assert len(backend.execute_calls) == 1
    assert len(backend.reconcile_calls) == 1
    events = (await store.read_events(job.job_id, 0, 100)).events
    assert events[-1].type == "job_failed"
    assert all(event.type not in {"message", "job_completed"} for event in events)


async def test_active_reconciliation_continues_on_current_generation_without_execute_replay():
    """An active provider operation stays observed under one fence until it completes."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    reference = ProviderReference(kind="thread", value="thread-active")
    failure = JobError(
        code="process_lost",
        message="Lost observation",
        retry_disposition="reconcile_required",
    )
    backend.queue_execute(
        RecordReferenceAction(reference),
        RaiseFailureAction(BackendFailure(failure, "reconcile_required")),
    )
    backend.queue_reconcile(
        ReturnReconciliationAction(ActiveReconciliationOutcome()),
        ReturnReconciliationAction(
            CompletedReconciliationOutcome(result=make_turn_result(message="Reattached"))
        ),
    )

    await make_worker(store, backend).run_once()

    assert len(backend.execute_calls) == 1
    assert len(backend.reconcile_calls) == 2
    attempts = await store.get_job_attempts(job.job_id)
    assert len(attempts) == 1
    terminal = await store.get_job(job.job_id)
    assert terminal is not None and terminal.state == "completed"


class RenewalTrackingStore(InMemoryJobStore):
    """Record that an active reconciliation observation keeps heartbeating its fence."""

    def __init__(self) -> None:
        super().__init__()
        self.renewed = asyncio.Event()

    async def renew_lease(self, token, lease_until):  # type: ignore[no-untyped-def]
        renewed = await super().renew_lease(token, lease_until)
        if renewed:
            self.renewed.set()
        return renewed


class ActiveThenGatedCompletionBackend(ScriptedBackend):
    """Report active once, then hold the next fenced observation before completion."""

    def __init__(self) -> None:
        super().__init__()
        self.second_observation = asyncio.Event()
        self.release = asyncio.Event()

    async def reconcile(self, provider_state, context: BackendExecutionContext):  # type: ignore[no-untyped-def]
        self.reconcile_calls.append((provider_state, context))
        if len(self.reconcile_calls) == 1:
            return ActiveReconciliationOutcome()
        self.second_observation.set()
        await self.release.wait()
        return CompletedReconciliationOutcome(result=make_turn_result(message="Observed"))


async def test_active_reconciliation_renews_current_lease_until_terminal_progression():
    """Active observation cannot return while leaving a phantom unrenewed lease behind."""
    store = RenewalTrackingStore()
    backend = ActiveThenGatedCompletionBackend()
    job = await admit(store)
    reference = ProviderReference(kind="thread", value="thread-current-owner")
    failure = JobError(
        code="process_lost",
        message="Lost observation",
        retry_disposition="reconcile_required",
    )
    backend.queue_execute(
        RecordReferenceAction(reference),
        RaiseFailureAction(BackendFailure(failure, "reconcile_required")),
    )
    policy = WorkerPolicy(
        lease_seconds=0.1,
        heartbeat_seconds=0.02,
        idle_poll_seconds=0.001,
        reconciliation_timeout_seconds=0.2,
    )
    worker_task = asyncio.create_task(make_worker(store, backend, policy=policy).run_once())
    await asyncio.wait_for(backend.second_observation.wait(), timeout=0.2)
    await asyncio.wait_for(store.renewed.wait(), timeout=0.2)

    current = await store.get_job(job.job_id)
    assert current is not None and current.state == "running"
    assert current.lease_owner_id == "worker-1"
    assert current.lease_generation == 1
    assert worker_task.done() is False

    backend.release.set()
    await worker_task
    terminal = await store.get_job(job.job_id)
    assert terminal is not None and terminal.state == "completed"


async def test_safe_retry_reexecutes_only_after_persisted_retry_schedule():
    """A backend-classified pre-reference failure creates a new attempt before replay."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    retry_policy = RetryPolicy(max_attempts=2, base_delay_seconds=0, max_delay_seconds=0)
    job = await admit(
        store,
        requested_config=RequestedExecutionConfig(
            explicit=ExecutionConfigValues(retry_policy=retry_policy)
        ),
    )
    retryable = JobError(
        code="provider_failed",
        message="Provider rejected before starting",
        retry_disposition="safe_to_retry",
        recoverable=True,
    )
    backend.queue_execute(
        RaiseFailureAction(BackendFailure(retryable, "safe_to_retry")),
        ReturnResultAction(make_turn_result(message="Second attempt")),
    )
    worker = make_worker(store, backend)

    await worker.run_once()
    scheduled = await store.get_job(job.job_id)
    assert scheduled is not None
    assert scheduled.state == "running"
    assert scheduled.lease_owner_id is None
    assert scheduled.retry_at is not None

    await worker.run_once()

    assert len(backend.execute_calls) == 2
    assert backend.execute_calls[1][1].attempt.phase == "executing"
    attempts = await store.get_job_attempts(job.job_id)
    assert len(attempts) == 2
    assert attempts[0].retry_classification == "safe_to_retry"


async def test_safe_retry_classification_cannot_replay_after_provider_reference():
    """A provider identity upgrades even a safe classification to reconciliation-only handling."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    reference = ProviderReference(kind="thread", value="started-already")
    retryable = JobError(
        code="provider_failed",
        message="Transport failed",
        retry_disposition="safe_to_retry",
        recoverable=True,
    )
    backend.queue_execute(
        RecordReferenceAction(reference),
        RaiseFailureAction(BackendFailure(retryable, "safe_to_retry")),
    )
    backend.queue_reconcile(
        ReturnReconciliationAction(
            CompletedReconciliationOutcome(result=make_turn_result(message="Recovered"))
        )
    )

    await make_worker(store, backend).run_once()

    assert len(backend.execute_calls) == 1
    assert len(backend.reconcile_calls) == 1
    terminal = await store.get_job(job.job_id)
    assert terminal is not None and terminal.state == "completed"


def test_worker_policy_defaults_keep_heartbeat_safely_inside_lease():
    """Default production timing retains more than two heartbeat opportunities."""
    policy = WorkerPolicy()
    assert policy.heartbeat_seconds * 2 < policy.lease_seconds
