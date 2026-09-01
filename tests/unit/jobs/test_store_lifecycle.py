"""Shared behavioral contract for every durable job-store implementation."""

import asyncio
from datetime import timedelta

import pytest

from nexus_mcp.core import (
    DiagnosticsOperation,
    InputAlreadyResolvedError,
    InputNotFoundError,
    PermissionResponse,
    ProviderReference,
    ResolvedExecutionConfig,
    StaleLeaseError,
)
from nexus_mcp.jobs.store import (
    CancelledTerminalOutcome,
    FailedTerminalOutcome,
    JobAccessFilter,
    JobQuery,
    ResolveInputCommand,
    SucceededTerminalOutcome,
)
from tests.fixtures import make_job_error, make_pending_permission, make_turn_result
from tests.unit.jobs._store_contract_support import (
    LEASE_UNTIL,
    NOW,
    OLD,
    make_cancel_job_command,
    make_create_job_command,
    make_event,
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

    await job_store.mark_running(
        claimed.token,
        (),
        event=make_event("job_started"),
    )
    await job_store.terminalize(
        claimed.token,
        SucceededTerminalOutcome(result=make_turn_result(), completed_at=OLD),
        event=make_event("job_completed"),
    )

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
    attempts = await job_store.get_job_attempts(created.handle.job_id)
    assert attempts[0].lease_expires_at == OLD
    assert attempts[0].heartbeat_at is not None
    assert attempts[0].ended_at is not None
    assert attempts[1].lease_expires_at == LEASE_UNTIL
    assert attempts[1].heartbeat_at is not None
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
    renewed_until = LEASE_UNTIL + timedelta(minutes=10)
    assert await job_store.renew_lease(current.token, renewed_until) is True
    attempts = await job_store.get_job_attempts(current.job.job_id)
    assert attempts[-1].lease_expires_at == renewed_until
    assert attempts[-1].heartbeat_at is not None
    assert current.attempt.heartbeat_at is not None
    assert attempts[-1].heartbeat_at >= current.attempt.heartbeat_at


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

    visible = await job_store.get_control_snapshot(claimed.token)
    assert visible.unresolved_inputs == ()
    assert len(visible.resolved_inputs) == 1
    assert visible.resolved_inputs[0].input_id == pending.input_id
    assert visible.resolved_inputs[0].response == command.response

    with pytest.raises(InputAlreadyResolvedError):
        await job_store.resolve_input(
            command.model_copy(update={"response": PermissionResponse(granted=[])})
        )

    await job_store.mark_running(
        claimed.token,
        (pending.input_id,),
        event=make_event("job_started", resumed=True),
    )
    replay_after_running = await job_store.resolve_input(command)
    assert replay_after_running.replayed is True
    with pytest.raises(InputAlreadyResolvedError):
        await job_store.resolve_input(
            command.model_copy(update={"response": PermissionResponse(granted=[])})
        )
    after = await job_store.get_control_snapshot(claimed.token)
    assert after.state == "running"
    events = await job_store.read_events(created.handle.job_id, 0, 20)
    assert [event.type for event in events.events].count("input_resolved") == 1


async def test_control_snapshot_atomically_partitions_multiple_input_responses(job_store):
    """A worker sees cross-task responses without trusting process-local delivery state."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-control", LEASE_UNTIL, event=make_event())
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    first = make_pending_permission(
        input_id="input-first", job_id=created.handle.job_id, created_at=NOW
    )
    second = make_pending_permission(
        input_id="input-second", job_id=created.handle.job_id, created_at=NOW
    )
    await job_store.mark_input_required(
        claimed.token,
        (first, second),
        event=make_event("input_required"),
    )
    await job_store.resolve_input(
        ResolveInputCommand(
            job_id=created.handle.job_id,
            input_id=second.input_id,
            response=PermissionResponse(granted=[]),
            resolved_at=NOW,
            event=make_event("input_resolved", input_id=second.input_id),
        )
    )

    snapshot = await job_store.get_control_snapshot(claimed.token)

    assert snapshot.unresolved_inputs == (first,)
    assert [item.input_id for item in snapshot.resolved_inputs] == [second.input_id]
    assert snapshot.resolved_inputs[0].response == PermissionResponse(granted=[])


async def test_control_snapshot_rejects_stale_fence_after_response_commit(job_store):
    """A resolved response cannot be observed through an obsolete lease generation."""
    created = await job_store.create_job(make_create_job_command())
    first_claim = await job_store.claim_next("worker-old", LEASE_UNTIL, event=make_event())
    assert first_claim is not None
    await job_store.mark_running(first_claim.token, (), event=make_event("job_started"))
    pending = make_pending_permission(job_id=created.handle.job_id, created_at=NOW)
    await job_store.mark_input_required(
        first_claim.token,
        (pending,),
        event=make_event("input_required"),
    )
    await job_store.resolve_input(
        ResolveInputCommand(
            job_id=created.handle.job_id,
            input_id=pending.input_id,
            response=PermissionResponse(granted=[]),
            resolved_at=NOW,
            event=make_event("input_resolved"),
        )
    )
    assert await job_store.renew_lease(first_claim.token, OLD) is True
    replacement = await job_store.claim_next("worker-new", LEASE_UNTIL, event=make_event())
    assert replacement is not None

    with pytest.raises(StaleLeaseError):
        await job_store.get_control_snapshot(first_claim.token)


async def test_resolved_input_replay_and_conflict_survive_terminal_state(job_store):
    """Stored response identity remains authoritative after the job terminalizes."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next(
        "worker-input-terminal", LEASE_UNTIL, event=make_event("progress")
    )
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    pending = make_pending_permission(job_id=created.handle.job_id, created_at=NOW)
    await job_store.mark_input_required(
        claimed.token,
        (pending,),
        event=make_event("input_required", input_id=pending.input_id),
    )
    command = ResolveInputCommand(
        job_id=created.handle.job_id,
        input_id=pending.input_id,
        response=PermissionResponse(granted=["network:api.example.com"]),
        resolved_at=NOW,
        event=make_event("input_resolved", input_id=pending.input_id),
    )
    await job_store.resolve_input(command)
    await job_store.mark_running(
        claimed.token,
        (pending.input_id,),
        event=make_event("job_started", resumed=True),
    )
    await job_store.terminalize(
        claimed.token,
        SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
        event=make_event("job_completed"),
    )
    before = await job_store.read_events(created.handle.job_id, 0, 20)

    replay = await job_store.resolve_input(command)
    with pytest.raises(InputAlreadyResolvedError):
        await job_store.resolve_input(
            command.model_copy(update={"response": PermissionResponse(granted=[])})
        )

    after = await job_store.read_events(created.handle.job_id, 0, 20)
    assert replay.replayed is True
    assert after.events == before.events
    assert [event.type for event in after.events].count("input_resolved") == 1


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
    assert second.attempt.phase == "executing"
    assert job is not None
    assert job.state == "running"
    assert second.token.generation == 2
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    assert attempts[0].reconciliation_classification == error.code
    assert attempts[0].retry_classification == error.retry_disposition
    assert attempts[0].lease_expires_at == LEASE_UNTIL
    assert attempts[0].heartbeat_at is not None


async def test_reconciliation_classification_survives_terminal_attempt_closure(job_store):
    """Terminal failure preserves why reconciliation began without inventing retry metadata."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    reconciliation_error = make_job_error(code="outcome_unknown")
    terminal_error = make_job_error(code="provider_failed")
    await job_store.mark_reconciling(
        claimed.token,
        reconciliation_error,
        event=make_event("reconciliation"),
    )

    await job_store.terminalize(
        claimed.token,
        FailedTerminalOutcome(error=terminal_error, completed_at=NOW),
        event=make_event("job_failed"),
    )

    attempt = (await job_store.get_job_attempts(created.handle.job_id))[-1]
    assert attempt.reconciliation_classification == reconciliation_error.code
    assert attempt.retry_classification is None
    assert attempt.error_code == terminal_error.code
    assert attempt.lease_expires_at == LEASE_UNTIL
    assert attempt.heartbeat_at is not None


async def test_active_cancellation_is_idempotent_and_completion_may_win(job_store):
    """Cancellation intent does not reopen or override a backend completion already committing."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-1", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    command = make_cancel_job_command(created.handle.job_id)

    first = await job_store.request_cancel(command)
    replay = await job_store.request_cancel(command)
    terminal = await job_store.terminalize(
        claimed.token,
        SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
        event=make_event("job_completed"),
    )
    after_terminal = await job_store.request_cancel(command)

    assert first.cancel_requested is True
    assert first.event_committed is True
    assert replay.cancel_requested is True
    assert replay.state == first.state
    assert replay.event_committed is False
    assert terminal.state == "completed"
    assert after_terminal.state == "completed"
    assert after_terminal.event_committed is False
    events = await job_store.read_events(created.handle.job_id, 0, 20)
    assert [event.type for event in events.events].count("cancel_requested") == 1


async def test_queued_cancellation_ignores_active_backend_capability(job_store):
    """The atomic queued branch terminalizes even when active interruption is unavailable."""
    created = await job_store.create_job(make_create_job_command())

    receipt = await job_store.request_cancel(
        make_cancel_job_command(
            created.handle.job_id,
            active_cancellation_allowed=False,
        )
    )

    assert receipt.state == "cancelled"
    assert receipt.completed_immediately is True
    assert receipt.event_committed is True
    events = await job_store.read_events(created.handle.job_id, 0, 20)
    assert events.events[-1].type == "job_cancelled"


async def test_queued_to_running_race_refuses_unsupported_active_cancellation(job_store):
    """A state change before the locked decision cannot persist unsupported active intent."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next("worker-race", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    before = await job_store.read_events(created.handle.job_id, 0, 20)

    receipt = await job_store.request_cancel(
        make_cancel_job_command(
            created.handle.job_id,
            active_cancellation_allowed=False,
        )
    )

    stored = await job_store.get_job(created.handle.job_id)
    after = await job_store.read_events(created.handle.job_id, 0, 20)
    assert stored is not None and stored.cancel_requested_at is None
    assert receipt.state == "running"
    assert receipt.cancel_requested is False
    assert receipt.event_committed is False
    assert after.events == before.events


async def test_concurrent_active_cancel_commits_exactly_one_intent_event(job_store):
    """Concurrent callers receive receipt truth for the single winning event transaction."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next(
        "worker-concurrent", LEASE_UNTIL, event=make_event("progress")
    )
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    command = make_cancel_job_command(created.handle.job_id)

    receipts = await asyncio.gather(
        job_store.request_cancel(command),
        job_store.request_cancel(command),
    )

    assert sum(receipt.event_committed for receipt in receipts) == 1
    assert all(receipt.cancel_requested for receipt in receipts)
    events = await job_store.read_events(created.handle.job_id, 0, 20)
    assert [event.type for event in events.events].count("cancel_requested") == 1


async def test_terminal_state_wins_without_committing_cancel_event(job_store):
    """A terminal job returns a no-op receipt even when active cancellation is disallowed."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next(
        "worker-terminal", LEASE_UNTIL, event=make_event("progress")
    )
    assert claimed is not None
    await job_store.mark_running(claimed.token, (), event=make_event("job_started"))
    await job_store.terminalize(
        claimed.token,
        SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
        event=make_event("job_completed"),
    )
    before = await job_store.read_events(created.handle.job_id, 0, 20)

    receipt = await job_store.request_cancel(
        make_cancel_job_command(
            created.handle.job_id,
            active_cancellation_allowed=False,
        )
    )

    after = await job_store.read_events(created.handle.job_id, 0, 20)
    assert receipt.state == "completed"
    assert receipt.event_committed is False
    assert after.events == before.events


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
        assert attempts[-1].retry_classification is None

    event_count = len((await job_store.read_events(created.handle.job_id, 0, 20)).events)
    receipt = await job_store.request_cancel(make_cancel_job_command(created.handle.job_id))
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
