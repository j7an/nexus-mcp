"""Shared behavioral contract for every durable job-store implementation."""

from datetime import UTC, datetime, timedelta

import pytest

from nexus_mcp.core import (
    BackendEvent,
    DiagnosticsOperation,
    DiagnosticsResult,
    PermissionResponse,
    ProviderReference,
)
from nexus_mcp.jobs.store import (
    CancelledTerminalOutcome,
    PrunePolicy,
    ResolveInputCommand,
    RuntimeLeaseBusyError,
    SucceededTerminalOutcome,
)
from tests.fixtures import make_pending_permission, make_turn_result
from tests.unit.jobs._store_contract_support import (
    LEASE_UNTIL,
    NOW,
    OLD,
    make_create_job_command,
    make_event,
)


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
            queued_event=BackendEvent(type="job_queued", occurred_at=OLD),
        )
    )
    unresolved_claim = await job_store.claim_next(
        "worker-1", LEASE_UNTIL, event=BackendEvent(type="progress", occurred_at=OLD)
    )
    assert unresolved_claim is not None
    await job_store.mark_running(
        unresolved_claim.token,
        (),
        event=BackendEvent(type="job_started", occurred_at=OLD),
    )
    unresolved_input = make_pending_permission(
        job_id=unresolved_job.handle.job_id,
        created_at=NOW,
    )
    await job_store.mark_input_required(
        unresolved_claim.token,
        (unresolved_input,),
        event=BackendEvent(type="input_required", occurred_at=OLD),
    )
    await job_store.terminalize(
        unresolved_claim.token,
        CancelledTerminalOutcome(completed_at=OLD),
        event=BackendEvent(type="job_cancelled", occurred_at=OLD),
    )

    resolved_job = await job_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            queued_event=BackendEvent(type="job_queued", occurred_at=OLD),
        )
    )
    resolved_claim = await job_store.claim_next(
        "worker-2", LEASE_UNTIL, event=BackendEvent(type="progress", occurred_at=OLD)
    )
    assert resolved_claim is not None
    await job_store.mark_running(
        resolved_claim.token,
        (),
        event=BackendEvent(type="job_started", occurred_at=OLD),
    )
    resolved_input = make_pending_permission(
        input_id="input-resolved",
        job_id=resolved_job.handle.job_id,
        created_at=NOW,
    )
    await job_store.mark_input_required(
        resolved_claim.token,
        (resolved_input,),
        event=BackendEvent(type="input_required", occurred_at=OLD),
    )
    await job_store.resolve_input(
        ResolveInputCommand(
            job_id=resolved_job.handle.job_id,
            input_id=resolved_input.input_id,
            response=PermissionResponse(granted=["network:api.example.com"]),
            resolved_at=NOW,
            event=BackendEvent(type="input_resolved", occurred_at=OLD),
        )
    )
    await job_store.terminalize(
        resolved_claim.token,
        CancelledTerminalOutcome(completed_at=OLD),
        event=BackendEvent(type="job_cancelled", occurred_at=OLD),
    )

    result = await job_store.prune(
        PrunePolicy(terminal_job_before=NOW, event_before=NOW),
        now=NOW,
    )

    assert result.terminal_jobs_deleted == 1
    assert await job_store.get_job(unresolved_job.handle.job_id) is not None
    assert await job_store.get_job(resolved_job.handle.job_id) is None
    unresolved_events = await job_store.read_events(unresolved_job.handle.job_id, 0, 10)
    assert [event.sequence for event in unresolved_events.events] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("state", ["queued", "running", "input_required"])
async def test_event_prune_retains_all_nonterminal_history(job_store, state: str):
    """Event retention never removes the evidence needed to recover active work."""
    created = await job_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            queued_event=BackendEvent(type="job_queued", occurred_at=OLD),
        )
    )
    expected_sequences = [1]
    if state != "queued":
        claimed = await job_store.claim_next(
            "worker-active",
            LEASE_UNTIL,
            event=BackendEvent(type="progress", occurred_at=OLD),
        )
        assert claimed is not None
        await job_store.mark_running(
            claimed.token,
            (),
            event=BackendEvent(type="job_started", occurred_at=OLD),
        )
        expected_sequences.extend((2, 3))
        if state == "input_required":
            await job_store.mark_input_required(
                claimed.token,
                (make_pending_permission(job_id=created.handle.job_id, created_at=OLD),),
                event=BackendEvent(type="input_required", occurred_at=OLD),
            )
            expected_sequences.append(4)

    result = await job_store.prune(
        PrunePolicy(event_before=datetime(2025, 6, 1, tzinfo=UTC)),
        now=NOW,
    )
    retained = await job_store.read_events(created.handle.job_id, 0, 10)

    assert result.events_deleted == 0
    assert [event.sequence for event in retained.events] == expected_sequences
    assert retained.latest_sequence == expected_sequences[-1]


async def test_event_prune_keeps_latest_sequence_watermark_for_eligible_terminal_job(job_store):
    """Pruned terminal detail retains the durable high-water sequence for status cursors."""
    created = await job_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            queued_event=BackendEvent(type="job_queued", occurred_at=OLD),
        )
    )
    claimed = await job_store.claim_next(
        "worker-terminal",
        LEASE_UNTIL,
        event=BackendEvent(type="progress", occurred_at=OLD),
    )
    assert claimed is not None
    await job_store.mark_running(
        claimed.token,
        (),
        event=BackendEvent(type="job_started", occurred_at=OLD),
    )
    await job_store.terminalize(
        claimed.token,
        SucceededTerminalOutcome(result=DiagnosticsResult(available=True), completed_at=OLD),
        event=BackendEvent(type="job_completed", occurred_at=OLD),
    )

    result = await job_store.prune(
        PrunePolicy(event_before=datetime(2025, 6, 1, tzinfo=UTC)),
        now=NOW,
    )
    retained = await job_store.read_events(created.handle.job_id, 0, 10)

    assert result.terminal_jobs_deleted == 0
    assert result.events_deleted == 4
    assert retained.events == ()
    assert retained.latest_sequence == 4


async def test_event_prune_retains_terminal_history_with_session_provider_reference(job_store):
    """A retained session's provider identity keeps its terminal reconciliation history."""
    created = await job_store.create_job(
        make_create_job_command(
            session_id="referenced-session",
            queued_event=BackendEvent(type="job_queued", occurred_at=OLD),
        )
    )
    claimed = await job_store.claim_next(
        "worker-reference",
        LEASE_UNTIL,
        event=BackendEvent(type="progress", occurred_at=OLD),
    )
    assert claimed is not None
    await job_store.mark_running(
        claimed.token,
        (),
        event=BackendEvent(type="job_started", occurred_at=OLD),
    )
    await job_store.record_provider_reference(
        claimed.token,
        ProviderReference(kind="thread", value="retained-thread"),
    )
    await job_store.terminalize(
        claimed.token,
        SucceededTerminalOutcome(result=make_turn_result(), completed_at=OLD),
        event=BackendEvent(type="job_completed", occurred_at=OLD),
    )

    result = await job_store.prune(
        PrunePolicy(event_before=datetime(2025, 6, 1, tzinfo=UTC)),
        now=NOW,
    )
    retained = await job_store.read_events(created.handle.job_id, 0, 10)

    assert result.events_deleted == 0
    assert [event.sequence for event in retained.events] == [1, 2, 3, 4]
    assert retained.latest_sequence == 4


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
    assert result.events_deleted == 4
    assert result.raw_diagnostics_deleted == 0
    assert await job_store.get_job(terminal_created.handle.job_id) is None
    assert await job_store.get_job(active.handle.job_id) is not None

    active_claim = await job_store.claim_next(
        "worker-2", LEASE_UNTIL, event=make_event("progress", after_prune=True)
    )
    assert active_claim is not None
    retained = await job_store.read_events(active.handle.job_id, after_sequence=0, limit=10)
    assert [event.sequence for event in retained.events] == [1, 2]
    assert retained.latest_sequence == 2
