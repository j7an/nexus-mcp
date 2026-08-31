"""Race, control, timeout, output, and lifecycle tests for durable workers."""

import asyncio
from datetime import datetime

import pytest

from nexus_mcp.backends import (
    BackendExecutionContext,
    CancelRequested,
    CompletedReconciliationOutcome,
    InputResolved,
    LeaseLost,
    RuntimeShutdown,
)
from nexus_mcp.backends.manager import BackendManager
from nexus_mcp.core import (
    AgentOperation,
    BackendEvent,
    ExecutionConfigValues,
    JobError,
    PermissionResponse,
    ProviderReference,
    QuestionRequest,
    QuestionResponse,
    RequestedExecutionConfig,
    StaleLeaseError,
)
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.store import CancelJobCommand, LeaseToken, ResolveInputCommand
from nexus_mcp.jobs.worker import WorkerPolicy, WorkerPool
from tests.fixtures import make_pending_permission, make_turn_result
from tests.job_fakes import (
    EmitOutputAction,
    InMemoryJobStore,
    RequestInputAction,
    ReturnReconciliationAction,
    ReturnResultAction,
    ScriptedBackend,
)
from tests.unit.jobs.test_worker import admit, make_worker


async def wait_until(predicate, *, timeout: float = 1.0) -> None:  # type: ignore[no-untyped-def]
    """Yield until one deterministic asynchronous test condition becomes true."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def cancel_command(job_id: str) -> CancelJobCommand:
    """Build a truthful atomic active-or-queued cancellation command."""
    return CancelJobCommand(
        job_id=job_id,
        active_cancellation_allowed=True,
        queued_event=BackendEvent(type="job_cancelled"),
        active_event=BackendEvent(type="cancel_requested"),
    )


async def test_request_input_flushes_output_and_returns_cross_task_response():
    """Input resumes from the committed response rather than notifier-local payload state."""
    store = InMemoryJobStore()
    notifier = EventNotifier()
    backend = ScriptedBackend()
    job = await admit(store)
    request = QuestionRequest(prompt="Continue?", choices=("yes", "no"), allow_free_text=False)
    backend.queue_execute(
        EmitOutputAction("before "),
        EmitOutputAction("input"),
        RequestInputAction(request),
        ReturnResultAction(make_turn_result(message="continued")),
    )
    worker_task = asyncio.create_task(make_worker(store, backend, notifier).run_once())

    await wait_until(lambda: bool(backend.input_requests))
    pending = await store.get_pending_inputs(job.job_id)
    assert len(pending) == 1
    response = QuestionResponse(answer="yes")
    await store.resolve_input(
        ResolveInputCommand(
            job_id=job.job_id,
            input_id=pending[0].input_id,
            response=response,
            event=BackendEvent(type="input_resolved", payload={"input_id": pending[0].input_id}),
        )
    )
    notifier.notify()
    await worker_task

    assert backend.input_responses == [response]
    events = (await store.read_events(job.job_id, 0, 100)).events
    types = [event.type for event in events]
    input_index = types.index("input_required")
    assert types[input_index - 1] == "message"
    assert dict(events[input_index - 1].payload) == {"text": "before input"}


class MultiControlBackend(ScriptedBackend):
    """Backend that observes two persisted control responses in provider order."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.signals: list[object] = []

    async def execute(self, operation: AgentOperation, context: BackendExecutionContext):
        self.execute_calls.append((operation, context))
        self.started.set()
        self.signals.append(await context.wait_for_control())
        self.signals.append(await context.wait_for_control())
        return make_turn_result(message="controls observed")


async def test_wait_for_control_observes_multiple_persisted_responses_once_each():
    """Notifier wakes are hints while fenced snapshots define exact resolved input identities."""
    store = InMemoryJobStore()
    notifier = EventNotifier()
    backend = MultiControlBackend()
    job = await admit(store)
    worker_task = asyncio.create_task(make_worker(store, backend, notifier).run_once())
    await backend.started.wait()
    running = await store.get_job(job.job_id)
    attempts = await store.get_job_attempts(job.job_id)
    assert running is not None
    token = LeaseToken(
        job_id=job.job_id,
        owner_id=str(running.lease_owner_id),
        generation=int(running.lease_generation),
        attempt_number=attempts[-1].attempt_number,
    )
    first = make_pending_permission(input_id="first", job_id=job.job_id)
    second = make_pending_permission(input_id="second", job_id=job.job_id)
    await store.mark_input_required(
        token,
        (first, second),
        event=BackendEvent(type="input_required"),
    )
    for expected_count, pending in enumerate((second, first), start=1):
        await store.resolve_input(
            ResolveInputCommand(
                job_id=job.job_id,
                input_id=pending.input_id,
                response=PermissionResponse(granted=[]),
                event=BackendEvent(type="input_resolved"),
            )
        )
        notifier.notify()
        await wait_until(lambda count=expected_count: len(backend.signals) >= count)
    await worker_task

    assert backend.signals == [InputResolved(input_id="second"), InputResolved(input_id="first")]


class CompletionGateBackend(ScriptedBackend):
    """Backend whose successful completion can be released after cancellation commits."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, operation: AgentOperation, context: BackendExecutionContext):
        self.execute_calls.append((operation, context))
        self.started.set()
        await self.release.wait()
        return make_turn_result(message="won the race")


async def test_completion_wins_race_with_active_cancellation_intent():
    """A committed provider completion remains truthful even when cancel intent arrived first."""
    store = InMemoryJobStore()
    notifier = EventNotifier()
    backend = CompletionGateBackend()
    job = await admit(store)
    worker_task = asyncio.create_task(make_worker(store, backend, notifier).run_once())
    await backend.started.wait()

    receipt = await store.request_cancel(cancel_command(job.job_id))
    notifier.notify()
    backend.release.set()
    await worker_task

    terminal = await store.get_job(job.job_id)
    assert receipt.cancel_requested is True
    assert terminal is not None and terminal.state == "completed"
    result = await store.get_job_result(job.job_id)
    assert result is not None and result.payload.message == "won the race"  # type: ignore[union-attr]


class CancelAwareBackend(ScriptedBackend):
    """Backend that reports which control signal selected provider cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.signal: object | None = None

    async def execute(self, operation: AgentOperation, context: BackendExecutionContext):
        self.execute_calls.append((operation, context))
        self.started.set()
        await context.emit_output_delta("before cancellation")
        self.signal = await context.wait_for_control()
        raise asyncio.CancelledError


async def test_active_cancellation_delivers_cancel_signal_and_terminalizes_cancelled():
    """User cancellation is distinct from lease loss and becomes terminal only after cleanup."""
    store = InMemoryJobStore()
    notifier = EventNotifier()
    backend = CancelAwareBackend()
    job = await admit(store)
    worker_task = asyncio.create_task(make_worker(store, backend, notifier).run_once())
    await backend.started.wait()

    await store.request_cancel(cancel_command(job.job_id))
    notifier.notify()
    await worker_task

    assert backend.signal == CancelRequested()
    terminal = await store.get_job(job.job_id)
    assert terminal is not None and terminal.state == "cancelled"
    events = (await store.read_events(job.job_id, 0, 100)).events
    assert [event.type for event in events[-2:]] == ["message", "job_cancelled"]
    assert dict(events[-2].payload) == {"text": "before cancellation"}


async def test_queued_cancellation_prevents_worker_execution():
    """A queued cancellation is already terminal and cannot later be claimed."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)

    receipt = await store.request_cancel(cancel_command(job.job_id))
    claimed = await make_worker(store, backend).run_once()

    assert receipt.completed_immediately is True
    assert claimed is False
    assert backend.execute_calls == []


class HangingBackend(ScriptedBackend):
    """Backend that makes timeout cancellation observable."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = asyncio.Event()

    async def execute(self, operation: AgentOperation, context: BackendExecutionContext):
        self.execute_calls.append((operation, context))
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


async def test_execution_timeout_is_a_truthful_terminal_timeout():
    """A pre-reference timeout fails durably with code=timeout and cancels local observation."""
    store = InMemoryJobStore()
    backend = HangingBackend()
    job = await admit(
        store,
        requested_config=RequestedExecutionConfig(
            explicit=ExecutionConfigValues(timeout_seconds=1)
        ),
    )

    await make_worker(store, backend).run_once()

    result = await store.get_job_result(job.job_id)
    assert isinstance(result, JobError)
    assert result.code == "timeout"
    assert backend.cancelled.is_set()


class ReferencedHangingBackend(HangingBackend):
    """Backend that persists provider identity before local observation times out."""

    async def execute(self, operation: AgentOperation, context: BackendExecutionContext):
        self.execute_calls.append((operation, context))
        await context.record_provider_reference(
            ProviderReference(kind="thread", value="timeout-thread")
        )
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


async def test_timeout_after_provider_reference_reconciles_without_replay():
    """Timeout uncertainty after provider start imports completion instead of executing twice."""
    store = InMemoryJobStore()
    backend = ReferencedHangingBackend()
    job = await admit(
        store,
        requested_config=RequestedExecutionConfig(
            explicit=ExecutionConfigValues(timeout_seconds=1)
        ),
    )
    backend.queue_reconcile(
        ReturnReconciliationAction(
            CompletedReconciliationOutcome(result=make_turn_result(message="after timeout"))
        )
    )

    await make_worker(store, backend).run_once()

    assert len(backend.execute_calls) == 1
    assert len(backend.reconcile_calls) == 1
    result = await store.get_job_result(job.job_id)
    assert result is not None and result.payload.message == "after timeout"  # type: ignore[union-attr]


class LeaseLosingStore(InMemoryJobStore):
    """Store that rejects the first heartbeat while leaving old state inspectable."""

    def __init__(self) -> None:
        super().__init__()
        self.heartbeat_attempted = asyncio.Event()

    async def renew_lease(self, token: LeaseToken, lease_until: datetime) -> bool:
        self.heartbeat_attempted.set()
        return False


class HeartbeatErrorStore(LeaseLosingStore):
    """Store whose renewal error makes continued ownership uncertain."""

    async def renew_lease(self, token: LeaseToken, lease_until: datetime) -> bool:
        self.heartbeat_attempted.set()
        raise RuntimeError("heartbeat storage unavailable")


class LeaseAwareBackend(ScriptedBackend):
    """Backend that distinguishes lease loss from user cancellation."""

    def __init__(self) -> None:
        super().__init__()
        self.signal: object | None = None

    async def execute(self, operation: AgentOperation, context: BackendExecutionContext):
        self.execute_calls.append((operation, context))
        self.signal = await context.wait_for_control()
        await asyncio.sleep(0)
        return make_turn_result(message="must not commit")


async def test_lease_loss_detaches_callbacks_without_false_terminal_write():
    """A stale generation observes lease_lost and cannot commit output or a terminal result."""
    store = LeaseLosingStore()
    backend = LeaseAwareBackend()
    job = await admit(store)
    policy = WorkerPolicy(
        lease_seconds=0.05,
        heartbeat_seconds=0.01,
        idle_poll_seconds=0.01,
        reconciliation_timeout_seconds=0.05,
    )
    worker = make_worker(store, backend, policy=policy)

    await worker.run_once()

    assert store.heartbeat_attempted.is_set()
    assert backend.signal == LeaseLost()
    saved_context = backend.execute_calls[0][1]
    with pytest.raises(StaleLeaseError):
        await saved_context.emit(BackendEvent(type="progress"))
    persisted = await store.get_job(job.job_id)
    assert persisted is not None and persisted.state == "running"
    assert await store.get_job_result(job.job_id) is None
    events = (await store.read_events(job.job_id, 0, 100)).events
    assert all(
        event.type not in {"job_completed", "job_failed", "job_cancelled"} for event in events
    )


async def test_heartbeat_error_detaches_as_lease_loss_instead_of_hanging():
    """A renewal error cannot leave callbacks authoritative after lease ownership is uncertain."""
    store = HeartbeatErrorStore()
    backend = LeaseAwareBackend()
    await admit(store)
    policy = WorkerPolicy(
        lease_seconds=0.05,
        heartbeat_seconds=0.01,
        idle_poll_seconds=0.01,
        reconciliation_timeout_seconds=0.05,
    )

    async with asyncio.timeout(0.2):
        await make_worker(store, backend, policy=policy).run_once()

    assert backend.signal == LeaseLost()


async def test_output_deltas_are_bounded_coalesced_and_followed_by_complete_message():
    """High-frequency deltas do not create one journal row each or lose final output."""
    store = InMemoryJobStore()
    backend = ScriptedBackend()
    job = await admit(store)
    complete = "🙂alpha-" * 8
    backend.queue_execute(
        *(EmitOutputAction(character) for character in complete),
        ReturnResultAction(make_turn_result(message=complete)),
    )

    await make_worker(store, backend, output_chunk_bytes=12).run_once()

    messages = [
        event
        for event in (await store.read_events(job.job_id, 0, 100)).events
        if event.type == "message"
    ]
    deltas = [event for event in messages if not event.payload.get("final")]
    assert "".join(str(event.payload["text"]) for event in deltas) == complete
    assert all(len(str(event.payload["text"]).encode("utf-8")) <= 12 for event in deltas)
    assert len(deltas) < len(complete)
    assert dict(messages[-1].payload) == {"text": complete, "final": True}


async def test_worker_pool_runs_until_idle_and_stops_interruptible_loops():
    """Pool compatibility draining and background shutdown leave no worker loop running."""
    store = InMemoryJobStore()
    notifier = EventNotifier()
    backend = ScriptedBackend()
    job = await admit(store)
    backend.queue_execute(ReturnResultAction(make_turn_result()))
    pool = WorkerPool(
        store=store,
        backends=BackendManager([backend]),
        notifier=notifier,
        worker_count=2,
        worker_id_prefix="pool-test",
        retry_delay=lambda attempt, retry_after, policy: 0.0,
    )

    assert await pool.run_until_idle() == 1
    assert (await store.get_job(job.job_id)).state == "completed"  # type: ignore[union-attr]

    await pool.start()
    assert pool.running is True
    await pool.stop()
    assert pool.running is False


class ShutdownAwareBackend(ScriptedBackend):
    """Backend that observes pool shutdown after buffered output is flushed."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.signal: object | None = None

    async def execute(self, operation: AgentOperation, context: BackendExecutionContext):
        self.execute_calls.append((operation, context))
        await context.emit_output_delta("before shutdown")
        self.started.set()
        self.signal = await context.wait_for_control()
        await asyncio.Event().wait()


class BlockingRenewalStore(InMemoryJobStore):
    """Store whose in-flight lease renewal remains blocked until cancelled or released."""

    def __init__(self) -> None:
        super().__init__()
        self.renewal_started = asyncio.Event()
        self.release_renewal = asyncio.Event()
        self.renewal_cancelled = asyncio.Event()

    async def renew_lease(self, token: LeaseToken, lease_until: datetime) -> bool:
        self.renewal_started.set()
        try:
            await self.release_renewal.wait()
        except asyncio.CancelledError:
            self.renewal_cancelled.set()
            raise
        return await super().renew_lease(token, lease_until)


async def test_worker_pool_stop_flushes_and_delivers_runtime_shutdown():
    """Pool shutdown detaches shared observation without a false provider terminal state."""
    store = InMemoryJobStore()
    notifier = EventNotifier()
    backend = ShutdownAwareBackend()
    job = await admit(store)
    pool = WorkerPool(
        store=store,
        backends=BackendManager([backend]),
        notifier=notifier,
        policy=WorkerPolicy(idle_poll_seconds=0.01),
    )
    await pool.start()
    await backend.started.wait()

    await pool.stop()

    assert backend.signal == RuntimeShutdown()
    persisted = await store.get_job(job.job_id)
    assert persisted is not None and persisted.state == "running"
    events = (await store.read_events(job.job_id, 0, 100)).events
    assert dict([event for event in events if event.type == "message"][-1].payload) == {
        "text": "before shutdown"
    }


async def test_worker_pool_stop_cancels_and_awaits_blocked_heartbeat_renewal():
    """Pool shutdown cannot wait for an external lease-renewal operation to release."""
    store = BlockingRenewalStore()
    notifier = EventNotifier()
    backend = ShutdownAwareBackend()
    await admit(store)
    pool = WorkerPool(
        store=store,
        backends=BackendManager([backend]),
        notifier=notifier,
        policy=WorkerPolicy(
            lease_seconds=0.05,
            heartbeat_seconds=0.01,
            idle_poll_seconds=0.01,
            reconciliation_timeout_seconds=0.05,
        ),
    )
    await pool.start()
    await backend.started.wait()
    await store.renewal_started.wait()
    stop_task = asyncio.create_task(pool.stop())

    try:
        async with asyncio.timeout(0.1):
            await asyncio.shield(stop_task)
    except TimeoutError:
        store.release_renewal.set()
        await stop_task
        pytest.fail("pool.stop() waited for blocked heartbeat renewal")

    assert store.renewal_cancelled.is_set()
    assert pool.running is False


class AvailabilityGateBackend(ScriptedBackend):
    """Backend whose pre-context availability check remains interruptibly blocked."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def check_availability(self, workspace):  # type: ignore[no-untyped-def]
        self.started.set()
        await self.release.wait()
        return self.availability


async def test_worker_pool_stop_interrupts_pre_context_preparation():
    """Pool shutdown does not wait indefinitely for availability before context creation."""
    store = InMemoryJobStore()
    notifier = EventNotifier()
    backend = AvailabilityGateBackend()
    job = await admit(store)
    pool = WorkerPool(
        store=store,
        backends=BackendManager([backend]),
        notifier=notifier,
        policy=WorkerPolicy(idle_poll_seconds=0.01),
    )
    await pool.start()
    await backend.started.wait()
    stop_task = asyncio.create_task(pool.stop())

    try:
        async with asyncio.timeout(0.1):
            await asyncio.shield(stop_task)
    except TimeoutError:
        backend.release.set()
        await stop_task
        pytest.fail("pool.stop() did not interrupt pre-context preparation")

    persisted = await store.get_job(job.job_id)
    assert persisted is not None and persisted.state == "running"
    assert backend.execute_calls == []
