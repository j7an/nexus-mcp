"""Shared behavioral contract for every durable job-store implementation."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue, ValidationError

from nexus_mcp.core import (
    BackendEvent,
    DiagnosticsOperation,
    DiagnosticsResult,
    IdempotencyConflictError,
    InputAlreadyResolvedError,
    InputNotFoundError,
    JobEventType,
    PermissionResponse,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    SessionBusyError,
    SessionNotFoundError,
    StaleLeaseError,
    TurnOperation,
    Workspace,
    WorkspaceInvalidError,
    WorkspaceSelector,
)
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore
from nexus_mcp.jobs.store import (
    CancelJobCommand,
    CancelledTerminalOutcome,
    CreateJobCommand,
    FailedTerminalOutcome,
    JobAccessFilter,
    JobQuery,
    PrunePolicy,
    ResolveInputCommand,
    RuntimeLeaseBusyError,
    SucceededTerminalOutcome,
)
from tests.fixtures import make_job_error, make_pending_permission, make_turn_result
from tests.job_fakes import InMemoryJobStore

NOW = datetime(2026, 8, 30, 20, 0, tzinfo=UTC)
OLD = datetime(2025, 1, 1, tzinfo=UTC)
LEASE_UNTIL = datetime(2099, 1, 1, tzinfo=UTC)
WORKSPACE_PATH = Path(__file__).resolve().parents[3]


def make_event(event_type: JobEventType = "progress", **payload: JsonValue) -> BackendEvent:
    """Build one stable normalized event for store assertions."""
    return BackendEvent(type=event_type, payload=payload, occurred_at=NOW)


def make_create_job_command(**overrides: Any) -> CreateJobCommand:
    """Build a stable admitted turn request with a new durable session."""
    defaults: dict[str, object] = {
        "workspace": Workspace(
            workspace_id="ws-test",
            canonical_path=WORKSPACE_PATH,
            created_at=NOW,
            updated_at=NOW,
        ),
        "backend_id": "codex",
        "owner_id": "local:501",
        "access_policy": "private",
        "operation": TurnOperation(prompt="Inspect the workspace"),
        "requested_config": RequestedExecutionConfig(),
        "session_id": "session-test",
        "create_session": True,
        "command_family": "submit",
        "queued_event": make_event("job_queued", status="queued"),
    }
    return CreateJobCommand(**(defaults | overrides))


@pytest.fixture(params=["memory", "sqlite"], ids=["memory", "sqlite"])
async def admission_store(request: pytest.FixtureRequest, tmp_path: Path):
    """Run admission, identity, and query assertions against every implementation."""
    store = (
        SQLiteJobStore(tmp_path / "jobs.sqlite3")
        if request.param == "sqlite"
        else InMemoryJobStore()
    )
    await store.open()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture(params=["memory", "sqlite"], ids=["memory", "sqlite"])
async def job_store(request: pytest.FixtureRequest, tmp_path: Path):
    """Run the complete lifecycle contract against every store implementation."""
    store = (
        SQLiteJobStore(tmp_path / "jobs.sqlite3")
        if request.param == "sqlite"
        else InMemoryJobStore()
    )
    await store.open()
    try:
        yield store
    finally:
        await store.close()


def test_create_command_requires_session_for_non_diagnostics():
    """A non-diagnostic operation cannot enter storage without a session identity."""
    with pytest.raises(ValidationError):
        make_create_job_command(session_id=None, create_session=False)


def test_create_command_keeps_diagnostics_sessionless():
    """Diagnostics cannot accidentally acquire conversation semantics."""
    with pytest.raises(ValidationError):
        make_create_job_command(operation=DiagnosticsOperation())

    command = make_create_job_command(
        operation=DiagnosticsOperation(),
        session_id=None,
        create_session=False,
    )
    assert command.session_id is None


def test_create_session_requires_session_id():
    """Session creation cannot commit an identity-free session record."""
    with pytest.raises(ValidationError):
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=True,
        )


def test_source_checkpoint_requires_a_derived_source_session():
    """Provider checkpoints cannot be admitted without a session that owns them."""
    with pytest.raises(ValidationError):
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            source_checkpoint=(ProviderReference(kind="thread", value="thread-1"),),
        )


def test_create_command_exposes_derived_source_session():
    """Idempotency and checkpoint provenance use one explicit source-session derivation."""
    root = make_create_job_command(session_id="root", create_session=True)
    continuation = make_create_job_command(session_id="existing", create_session=False)
    child = make_create_job_command(
        session_id="child",
        create_session=True,
        parent_session_id="parent",
    )

    assert root.source_session_id is None
    assert continuation.source_session_id == "existing"
    assert child.source_session_id == "parent"


def test_job_query_requires_nonempty_states_and_bounded_limit():
    """List calls cannot become unbounded scans or meaningless state filters."""
    access = JobAccessFilter(principal_id="local:501")
    with pytest.raises(ValidationError):
        JobQuery(workspace_id="ws-test", access=access, states=frozenset(), limit=10)
    with pytest.raises(ValidationError):
        JobQuery(workspace_id="ws-test", access=access, states={"queued"}, limit=101)


def test_prune_policy_couples_raw_cutoff_and_positive_byte_cap():
    """Raw diagnostics cannot be pruned by age without an explicit byte ceiling."""
    with pytest.raises(ValidationError):
        PrunePolicy(raw_diagnostic_before=NOW)
    with pytest.raises(ValidationError):
        PrunePolicy(raw_diagnostic_max_bytes=1024)
    with pytest.raises(ValidationError):
        PrunePolicy(raw_diagnostic_before=NOW, raw_diagnostic_max_bytes=0)


async def test_create_job_replays_matching_idempotency_key(admission_store):
    """The same scoped request key returns its original handle without another event."""
    command = make_create_job_command(idempotency_key="request-1")
    first = await admission_store.create_job(command)
    second = await admission_store.create_job(command)

    assert second.handle == first.handle
    assert second.created is False
    page = await admission_store.read_events(first.handle.job_id, after_sequence=0, limit=10)
    assert [event.sequence for event in page.events] == [1]


async def test_idempotency_key_rejects_a_different_request(admission_store):
    """A key cannot alias two different immutable operation requests."""
    await admission_store.create_job(make_create_job_command(idempotency_key="request-1"))

    with pytest.raises(IdempotencyConflictError):
        await admission_store.create_job(
            make_create_job_command(
                idempotency_key="request-1",
                operation=TurnOperation(prompt="Do something else"),
            )
        )


async def test_idempotency_replay_after_terminal_returns_original_handle(job_store):
    """A late retry cannot reopen a terminal job or change its original admission handle."""
    command = make_create_job_command(idempotency_key="request-terminal")
    first = await job_store.create_job(command)
    await job_store.request_cancel(
        CancelJobCommand(
            job_id=first.handle.job_id,
            requested_at=NOW,
            event=make_event("job_cancelled"),
        )
    )

    replay = await job_store.create_job(command)

    assert replay.created is False
    assert replay.handle == first.handle
    stored = await job_store.get_job(first.handle.job_id)
    assert stored is not None and stored.state == "cancelled"


async def test_same_idempotency_key_is_independent_across_source_sessions(admission_store):
    """Continuation idempotency is scoped to the session from which work derives."""
    for session_id in ("source-a", "source-b"):
        await admission_store.create_job(
            make_create_job_command(session_id=session_id, idempotency_key=None)
        )

    first = await admission_store.create_job(
        make_create_job_command(
            session_id="child-a",
            parent_session_id="source-a",
            idempotency_key="same-key",
        )
    )
    second = await admission_store.create_job(
        make_create_job_command(
            session_id="child-b",
            parent_session_id="source-b",
            idempotency_key="same-key",
        )
    )

    assert first.created is True
    assert second.created is True
    assert second.handle.job_id != first.handle.job_id


async def test_second_nonterminal_job_for_session_is_rejected(admission_store):
    """One session cannot admit overlapping provider operations."""
    await admission_store.create_job(make_create_job_command(session_id="session-1"))

    with pytest.raises(SessionBusyError):
        await admission_store.create_job(make_create_job_command(session_id="session-1"))


async def test_terminal_session_accepts_a_new_job(job_store):
    """The per-session uniqueness fence releases only after terminal cancellation."""
    first = await job_store.create_job(make_create_job_command(session_id="session-1"))
    receipt = await job_store.request_cancel(
        CancelJobCommand(
            job_id=first.handle.job_id,
            requested_at=NOW,
            event=make_event("job_cancelled"),
        )
    )

    second = await job_store.create_job(make_create_job_command(session_id="session-1"))

    assert receipt.completed_immediately is True
    assert second.handle.job_id != first.handle.job_id


async def test_claimed_queued_cancellation_closes_the_attempt(job_store):
    """Immediate queued cancellation cannot leave its claiming attempt active."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next(
        "worker-1", LEASE_UNTIL, event=make_event("progress", action="claimed")
    )
    assert claimed is not None

    await job_store.request_cancel(
        CancelJobCommand(
            job_id=created.handle.job_id,
            requested_at=NOW,
            event=make_event("job_cancelled"),
        )
    )

    attempts = await job_store.get_job_attempts(created.handle.job_id)
    assert attempts[-1].phase == "finalizing"
    assert attempts[-1].ended_at == NOW


async def test_create_persists_workspace_session_and_queued_event(admission_store):
    """Admission atomically persists its durable identities and initial history."""
    command = make_create_job_command()
    created = await admission_store.create_job(command)

    assert (
        await admission_store.resolve_workspace(WorkspaceSelector(workspace_id="ws-test"))
        == command.workspace
    )
    assert (
        await admission_store.resolve_workspace(WorkspaceSelector(path=WORKSPACE_PATH))
        == command.workspace
    )
    session = await admission_store.get_session("session-test")
    assert session is not None
    assert session.workspace_id == "ws-test"
    events = await admission_store.read_events(created.handle.job_id, after_sequence=0, limit=10)
    assert [(event.sequence, event.type) for event in events.events] == [(1, "job_queued")]


async def test_create_round_trips_typed_job_and_session_snapshots(admission_store):
    """Admission reads reconstruct closed job and session models without untyped payloads."""
    command = make_create_job_command(idempotency_key="round-trip")
    created = await admission_store.create_job(command)

    job = await admission_store.get_job(created.handle.job_id)
    session = await admission_store.get_session("session-test")

    assert job is not None
    assert job.operation == command.operation
    assert job.requested_config == command.requested_config
    assert job.owner_id == command.owner_id
    assert job.access_policy == command.access_policy
    assert job.idempotency_key == command.idempotency_key
    assert job.source_checkpoint == command.source_checkpoint
    assert session is not None
    assert session.workspace_id == command.workspace.workspace_id
    assert session.backend_id == command.backend_id
    assert session.owner_id == command.owner_id
    assert session.access_policy == command.access_policy


async def test_failed_create_job_leaves_no_partial_workspace(admission_store):
    """An admission failure rolls back every identity that would have accompanied the job."""
    command = make_create_job_command(
        workspace=Workspace(
            workspace_id="ws-rollback",
            canonical_path=Path("/tmp/nexus-rollback"),
            created_at=NOW,
            updated_at=NOW,
        ),
        session_id="missing-session",
        create_session=False,
    )

    with pytest.raises(SessionNotFoundError):
        await admission_store.create_job(command)
    with pytest.raises(WorkspaceInvalidError):
        await admission_store.resolve_workspace(WorkspaceSelector(workspace_id="ws-rollback"))


async def test_source_checkpoint_requires_existing_source_before_any_mutation(admission_store):
    """A missing checkpoint source rolls back workspace, child session, job, event, and key."""
    workspace = Workspace(
        workspace_id="ws-missing-source",
        canonical_path=Path("/tmp/nexus-missing-source"),
        created_at=NOW,
        updated_at=NOW,
    )
    command = make_create_job_command(
        workspace=workspace,
        session_id="child-missing-source",
        parent_session_id="missing-source",
        source_checkpoint=(ProviderReference(kind="thread", value="thread-missing"),),
        idempotency_key="missing-source-key",
    )

    with pytest.raises(SessionNotFoundError):
        await admission_store.create_job(command)
    assert await admission_store.get_session("child-missing-source") is None
    with pytest.raises(WorkspaceInvalidError):
        await admission_store.resolve_workspace(
            WorkspaceSelector(workspace_id=workspace.workspace_id)
        )


async def test_source_checkpoint_must_belong_to_source_session_atomically(job_store):
    """A mismatched provider reference creates no child state or idempotency residue."""
    await job_store.create_job(make_create_job_command(session_id="source-session"))
    claimed = await job_store.claim_next(
        "worker-1", LEASE_UNTIL, event=make_event("progress", action="claimed")
    )
    assert claimed is not None
    owned = ProviderReference(kind="thread", value="thread-owned")
    await job_store.record_provider_reference(claimed.token, owned)
    mismatched = ProviderReference(kind="thread", value="thread-other")
    invalid = make_create_job_command(
        session_id="child-session",
        parent_session_id="source-session",
        source_checkpoint=(mismatched,),
        idempotency_key="child-key",
    )

    with pytest.raises(ValueError, match="source checkpoint"):
        await job_store.create_job(invalid)
    assert await job_store.get_session("child-session") is None

    valid = await job_store.create_job(invalid.model_copy(update={"source_checkpoint": (owned,)}))
    assert valid.created is True
    assert await job_store.get_provider_references(job_id=valid.handle.job_id) == (owned,)
    child = await job_store.get_session("child-session")
    assert child is not None
    assert child.provider_references == ()


async def test_claim_allocates_attempt_event_and_new_fencing_generation(job_store):
    """Reclaiming an expired lease fences the old attempt and appends exact history."""
    created = await job_store.create_job(make_create_job_command())
    first = await job_store.claim_next(
        "worker-1",
        OLD,
        event=make_event("progress", action="claimed"),
    )
    second = await job_store.claim_next(
        "worker-2",
        LEASE_UNTIL,
        event=make_event("reconciliation", action="reclaimed"),
    )

    assert first is not None
    assert second is not None
    assert first.job.job_id == created.handle.job_id
    assert first.token.generation == 1
    assert first.attempt.attempt_number == 1
    assert second.token.generation == 2
    assert second.attempt.attempt_number == 2
    assert second.attempt.phase == "reconciling"
    events = await job_store.read_events(created.handle.job_id, 0, 10)
    assert [event.sequence for event in events.events] == [1, 2, 3]


async def test_stale_job_lease_cannot_publish_worker_state(job_store):
    """An expired generation loses mutation authority as soon as a reclaim commits."""
    await job_store.create_job(make_create_job_command())
    stale = await job_store.claim_next("worker-1", OLD, event=make_event("progress"))
    current = await job_store.claim_next(
        "worker-2", LEASE_UNTIL, event=make_event("reconciliation")
    )
    assert stale is not None
    assert current is not None

    with pytest.raises(StaleLeaseError):
        await job_store.append_events(stale.token, (make_event("message"),))
    assert await job_store.renew_lease(stale.token, LEASE_UNTIL + timedelta(minutes=10)) is False
    assert await job_store.renew_lease(current.token, LEASE_UNTIL + timedelta(minutes=10)) is True


async def test_expired_active_job_is_reclaimed_for_reconciliation(job_store):
    """An expired running lease is reclaimed without replaying the admitted operation."""
    await job_store.create_job(make_create_job_command())
    first = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert first is not None
    await job_store.mark_running(first.token, (), event=make_event("job_started"))
    assert await job_store.renew_lease(first.token, OLD) is True

    reclaimed = await job_store.claim_next(
        "worker-2", LEASE_UNTIL, event=make_event("reconciliation")
    )

    assert reclaimed is not None
    assert reclaimed.token.generation == first.token.generation + 1
    assert reclaimed.attempt.phase == "reconciling"


async def test_worker_config_references_and_events_remain_fenced(job_store):
    """Current worker-owned facts persist without exposing a transaction primitive."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    config = ResolvedExecutionConfig(model="gpt-test", sources={"model": "provider"})
    reference = ProviderReference(kind="thread", value="thread-1")

    await job_store.store_resolved_config(claimed.token, config)
    await job_store.record_provider_reference(claimed.token, reference)
    committed = await job_store.append_events(
        claimed.token,
        (make_event("message", text="one"), make_event("usage", tokens=1)),
    )

    job = await job_store.get_job(created.handle.job_id)
    assert job is not None
    assert job.resolved_config == config
    assert await job_store.get_provider_references(job_id=job.job_id) == (reference,)
    assert await job_store.get_provider_references(session_id="session-test") == (reference,)
    assert [event.sequence for event in committed] == [3, 4]


async def test_input_lifecycle_is_atomic_validated_and_idempotent(job_store):
    """Input state and events advance once while response replays remain event-free."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    pending = make_pending_permission(job_id=created.handle.job_id, created_at=NOW)
    await job_store.mark_input_required(
        claimed.token,
        (pending,),
        event=make_event("input_required", input_id=pending.input_id),
    )

    before = await job_store.get_control_snapshot(claimed.token)
    assert before.state == "input_required"
    assert before.unresolved_inputs == (pending,)
    assert before.lease_generation == claimed.token.generation

    command = ResolveInputCommand(
        job_id=created.handle.job_id,
        input_id=pending.input_id,
        response=PermissionResponse(granted=["network:api.example.com"]),
        resolved_at=NOW,
        event=make_event("input_resolved", input_id=pending.input_id),
    )
    first = await job_store.resolve_input(command)
    replay = await job_store.resolve_input(command)
    assert first.replayed is False
    assert replay.replayed is True
    assert await job_store.get_pending_inputs(created.handle.job_id) == ()

    with pytest.raises(InputAlreadyResolvedError):
        await job_store.resolve_input(
            command.model_copy(update={"response": PermissionResponse(granted=[])})
        )

    await job_store.mark_running(
        claimed.token,
        (pending.input_id,),
        event=make_event("job_started", resumed=True),
    )
    after = await job_store.get_control_snapshot(claimed.token)
    assert after.state == "running"
    events = await job_store.read_events(created.handle.job_id, 0, 20)
    assert [event.type for event in events.events].count("input_resolved") == 1


async def test_terminalization_wins_race_with_input_resolution(job_store):
    """A response arriving after terminal commit cannot mutate input state or append history."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    pending = make_pending_permission(job_id=created.handle.job_id, created_at=NOW)
    await job_store.mark_input_required(
        claimed.token,
        (pending,),
        event=make_event("input_required", input_id=pending.input_id),
    )
    await job_store.terminalize(
        claimed.token,
        CancelledTerminalOutcome(completed_at=NOW),
        event=make_event("job_cancelled"),
    )
    before = await job_store.read_events(created.handle.job_id, 0, 20)

    with pytest.raises(InputNotFoundError):
        await job_store.resolve_input(
            ResolveInputCommand(
                job_id=created.handle.job_id,
                input_id=pending.input_id,
                response=PermissionResponse(granted=["network:api.example.com"]),
                resolved_at=NOW,
                event=make_event("input_resolved"),
            )
        )

    after = await job_store.read_events(created.handle.job_id, 0, 20)
    assert after.events == before.events
    assert after.events[-1].type == "job_cancelled"
    assert await job_store.get_pending_inputs(created.handle.job_id) == (pending,)


async def test_reconciliation_and_retry_create_a_new_attempt_without_reopening_state(job_store):
    """Safe retry preserves the operation identity and fences a new attempt generation."""
    created = await job_store.create_job(make_create_job_command())
    first = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert first is not None
    await job_store.mark_running(first.token, (), event=make_event("job_started"))
    error = make_job_error(retry_disposition="safe_to_retry", recoverable=True)
    await job_store.mark_reconciling(
        first.token, error, event=make_event("reconciliation", state="checking")
    )
    await job_store.schedule_retry(
        first.token,
        OLD,
        error,
        event=make_event("retry_scheduled"),
    )

    second = await job_store.claim_next("worker-2", LEASE_UNTIL, event=make_event("reconciliation"))
    job = await job_store.get_job(created.handle.job_id)
    attempts = await job_store.get_job_attempts(created.handle.job_id)
    assert second is not None
    assert job is not None
    assert job.state == "running"
    assert second.token.generation == 2
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]


async def test_active_cancellation_is_idempotent_and_completion_may_win(job_store):
    """Cancellation intent does not reopen or override a backend completion already committing."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    command = CancelJobCommand(
        job_id=created.handle.job_id,
        requested_at=NOW,
        event=make_event("cancel_requested"),
    )

    first = await job_store.request_cancel(command)
    replay = await job_store.request_cancel(command)
    terminal = await job_store.terminalize(
        claimed.token,
        SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
        event=make_event("job_completed"),
    )
    after_terminal = await job_store.request_cancel(command)

    assert first.cancel_requested is True
    assert replay == first
    assert terminal.state == "completed"
    assert after_terminal.state == "completed"
    events = await job_store.read_events(created.handle.job_id, 0, 20)
    assert [event.type for event in events.events].count("cancel_requested") == 1


@pytest.mark.parametrize(
    ("outcome", "expected_state", "result_type"),
    [
        (
            SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
            "completed",
            "result",
        ),
        (FailedTerminalOutcome(error=make_job_error(), completed_at=NOW), "failed", "error"),
        (CancelledTerminalOutcome(completed_at=NOW), "cancelled", "none"),
    ],
)
async def test_terminalize_atomically_stores_each_outcome(
    job_store, outcome, expected_state: str, result_type: str
):
    """Each closed terminal variant writes its snapshot, result, and final event together."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))

    terminal = await job_store.terminalize(
        claimed.token,
        outcome,
        event=make_event(
            {"completed": "job_completed", "failed": "job_failed", "cancelled": "job_cancelled"}[
                expected_state
            ]
        ),
    )
    stored_result = await job_store.get_job_result(created.handle.job_id)

    assert terminal.state == expected_state
    if result_type == "result":
        assert stored_result is not None and stored_result.job_id == created.handle.job_id
    elif result_type == "error":
        assert stored_result == outcome.error
    else:
        assert stored_result is None
    attempts = await job_store.get_job_attempts(created.handle.job_id)
    assert attempts[-1].ended_at == outcome.completed_at
    if result_type == "error":
        assert attempts[-1].error_code == outcome.error.code
        assert attempts[-1].error_message == outcome.error.message

    event_count = len((await job_store.read_events(created.handle.job_id, 0, 20)).events)
    receipt = await job_store.request_cancel(
        CancelJobCommand(
            job_id=created.handle.job_id,
            requested_at=NOW,
            event=make_event("cancel_requested"),
        )
    )
    assert receipt.state == expected_state
    assert len((await job_store.read_events(created.handle.job_id, 0, 20)).events) == event_count


async def test_list_jobs_applies_access_order_and_opaque_cursor(admission_store):
    """Stored pages expose only caller-visible jobs in deterministic descending order."""
    own = await admission_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            owner_id="local:501",
        )
    )
    await admission_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            owner_id="local:999",
            access_policy="private",
        )
    )
    shared = await admission_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            owner_id="local:999",
            access_policy="workspace",
        )
    )
    query = JobQuery(
        workspace_id="ws-test",
        access=JobAccessFilter(principal_id="local:501", workspace_authorized=True),
        states={"queued"},
        limit=1,
    )

    first = await admission_store.list_jobs(query)
    second = await admission_store.list_jobs(query.model_copy(update={"cursor": first.next_cursor}))
    private = await admission_store.list_jobs(
        query.model_copy(
            update={
                "access": JobAccessFilter(principal_id="local:501"),
                "limit": 100,
                "cursor": None,
            }
        )
    )

    visible = (*first.jobs, *second.jobs)
    assert {job.job_id for job in visible} == {own.handle.job_id, shared.handle.job_id}
    assert [job.job_id for job in visible] == [
        job.job_id
        for job in sorted(visible, key=lambda job: (job.created_at, job.job_id), reverse=True)
    ]
    assert first.next_cursor is not None
    assert [job.job_id for job in private.jobs] == [own.handle.job_id]


async def test_event_pages_advance_from_committed_sequence(job_store):
    """Event cursors resume strictly after the caller's last committed sequence."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await job_store.append_events(
        claimed.token,
        (make_event("message", part=1), make_event("message", part=2)),
    )

    first = await job_store.read_events(created.handle.job_id, after_sequence=0, limit=2)
    second = await job_store.read_events(
        created.handle.job_id,
        after_sequence=first.next_after_sequence or 0,
        limit=2,
    )
    assert [event.sequence for event in first.events] == [1, 2]
    assert first.has_more is True
    assert first.next_after_sequence == 2
    assert [event.sequence for event in second.events] == [3, 4]
    assert second.has_more is False


async def test_runtime_lease_contention_generation_and_endpoint_fencing(job_store):
    """A live runtime owner excludes contenders and stale generations cannot renew or release."""
    far_future = datetime(2099, 1, 1, tzinfo=UTC)
    first = await job_store.acquire_runtime_lease("opencode:ws-test", "process-1", far_future)
    extended = await job_store.acquire_runtime_lease(
        "opencode:ws-test", "process-1", far_future + timedelta(minutes=5)
    )
    assert extended.generation == first.generation

    with pytest.raises(RuntimeLeaseBusyError):
        await job_store.acquire_runtime_lease("opencode:ws-test", "process-2", far_future)

    expired = await job_store.acquire_runtime_lease("opencode:expired", "process-1", OLD)
    current = await job_store.acquire_runtime_lease("opencode:expired", "process-2", far_future)
    assert current.generation == expired.generation + 1
    assert await job_store.renew_runtime_lease(expired, far_future) is False

    with_endpoint = extended.model_copy(update={"endpoint": "http://127.0.0.1:4096"})
    assert (
        await job_store.renew_runtime_lease(with_endpoint, far_future + timedelta(minutes=10))
        is True
    )
    await job_store.release_runtime_lease(current)
    assert (
        await job_store.renew_runtime_lease(with_endpoint, far_future + timedelta(minutes=15))
        is True
    )
    await job_store.release_runtime_lease(with_endpoint)
    replacement = await job_store.acquire_runtime_lease("opencode:ws-test", "process-3", far_future)
    assert replacement.generation == first.generation + 1

    abandoned = await job_store.acquire_runtime_lease("opencode:abandoned", "process-1", OLD)
    assert await job_store.renew_runtime_lease(abandoned, far_future) is False
    await job_store.release_runtime_lease(abandoned)
    fenced = await job_store.acquire_runtime_lease("opencode:abandoned", "process-2", far_future)
    assert fenced.generation == abandoned.generation + 1

    same_owner_old = await job_store.acquire_runtime_lease(
        "opencode:same-owner", "process-1", far_future
    )
    await job_store.release_runtime_lease(same_owner_old)
    same_owner_current = await job_store.acquire_runtime_lease(
        "opencode:same-owner", "process-1", far_future
    )
    assert same_owner_current.generation == same_owner_old.generation + 1
    assert await job_store.renew_runtime_lease(same_owner_old, far_future) is False
    await job_store.release_runtime_lease(same_owner_old)
    assert await job_store.renew_runtime_lease(same_owner_current, far_future) is True


async def test_prune_retains_terminal_jobs_with_unresolved_inputs(job_store):
    """Retention cannot erase a terminal snapshot while durable input remains unresolved."""
    unresolved_job = await job_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
        )
    )
    unresolved_claim = await job_store.claim_next(
        "worker-1", LEASE_UNTIL, event=make_event("progress")
    )
    assert unresolved_claim is not None
    await job_store.mark_running(unresolved_claim.token, (), event=make_event("job_started"))
    unresolved_input = make_pending_permission(
        job_id=unresolved_job.handle.job_id,
        created_at=NOW,
    )
    await job_store.mark_input_required(
        unresolved_claim.token,
        (unresolved_input,),
        event=make_event("input_required"),
    )
    await job_store.terminalize(
        unresolved_claim.token,
        CancelledTerminalOutcome(completed_at=OLD),
        event=make_event("job_cancelled"),
    )

    resolved_job = await job_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
        )
    )
    resolved_claim = await job_store.claim_next(
        "worker-2", LEASE_UNTIL, event=make_event("progress")
    )
    assert resolved_claim is not None
    await job_store.mark_running(resolved_claim.token, (), event=make_event("job_started"))
    resolved_input = make_pending_permission(
        input_id="input-resolved",
        job_id=resolved_job.handle.job_id,
        created_at=NOW,
    )
    await job_store.mark_input_required(
        resolved_claim.token,
        (resolved_input,),
        event=make_event("input_required"),
    )
    await job_store.resolve_input(
        ResolveInputCommand(
            job_id=resolved_job.handle.job_id,
            input_id=resolved_input.input_id,
            response=PermissionResponse(granted=["network:api.example.com"]),
            resolved_at=NOW,
            event=make_event("input_resolved"),
        )
    )
    await job_store.terminalize(
        resolved_claim.token,
        CancelledTerminalOutcome(completed_at=OLD),
        event=make_event("job_cancelled"),
    )

    result = await job_store.prune(PrunePolicy(terminal_job_before=NOW), now=NOW)

    assert result.terminal_jobs_deleted == 1
    assert await job_store.get_job(unresolved_job.handle.job_id) is not None
    assert await job_store.get_job(resolved_job.handle.job_id) is None


async def test_prune_removes_only_eligible_terminal_jobs_and_old_events(job_store):
    """Retention cutoffs remain independent and never delete active job snapshots."""
    terminal_created = await job_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            queued_event=BackendEvent(type="job_queued", occurred_at=OLD),
        )
    )
    terminal_claim = await job_store.claim_next(
        "worker-1",
        LEASE_UNTIL,
        event=BackendEvent(type="progress", occurred_at=OLD),
    )
    assert terminal_claim is not None
    await job_store.mark_running(
        terminal_claim.token, (), event=BackendEvent(type="job_started", occurred_at=OLD)
    )
    await job_store.terminalize(
        terminal_claim.token,
        SucceededTerminalOutcome(
            result=DiagnosticsResult(available=True),
            completed_at=OLD,
        ),
        event=BackendEvent(type="job_completed", occurred_at=OLD),
    )
    active = await job_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            queued_event=BackendEvent(type="job_queued", occurred_at=OLD),
        )
    )

    result = await job_store.prune(
        PrunePolicy(
            terminal_job_before=datetime(2025, 6, 1, tzinfo=UTC),
            event_before=datetime(2025, 6, 1, tzinfo=UTC),
        ),
        now=NOW,
    )

    assert result.terminal_jobs_deleted == 1
    assert result.events_deleted == 5
    assert result.raw_diagnostics_deleted == 0
    assert await job_store.get_job(terminal_created.handle.job_id) is None
    assert await job_store.get_job(active.handle.job_id) is not None

    active_claim = await job_store.claim_next(
        "worker-2", LEASE_UNTIL, event=make_event("progress", after_prune=True)
    )
    assert active_claim is not None
    retained = await job_store.read_events(active.handle.job_id, after_sequence=0, limit=10)
    assert [event.sequence for event in retained.events] == [2]
