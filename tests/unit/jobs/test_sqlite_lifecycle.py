"""SQLite-specific lifecycle fencing and transaction regressions."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nexus_mcp.core import BackendEvent, ProviderReference, StaleLeaseError
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore
from nexus_mcp.jobs.store import CancelledTerminalOutcome, PrunePolicy
from tests.fixtures import make_pending_permission
from tests.unit.jobs.test_store_contract import make_create_job_command, make_event

OLD = datetime(2025, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 8, 30, 20, 0, tzinfo=UTC)
LEASE_UNTIL = datetime(2099, 1, 1, tzinfo=UTC)


@pytest.fixture
async def sqlite_store(tmp_path: Path):
    """Open one private SQLite store for a lifecycle assertion."""
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    await store.open()
    try:
        yield store
    finally:
        await store.close()


async def test_stale_generation_cannot_terminalize(sqlite_store: SQLiteJobStore):
    """A reclaimed lease prevents its predecessor from publishing a terminal snapshot or event."""
    created = await sqlite_store.create_job(make_create_job_command())
    first = await sqlite_store.claim_next("worker-a", OLD, event=make_event("progress"))
    second = await sqlite_store.claim_next(
        "worker-b", LEASE_UNTIL, event=make_event("reconciliation")
    )
    assert first is not None
    assert second is not None

    with pytest.raises(StaleLeaseError):
        await sqlite_store.terminalize(
            first.token,
            CancelledTerminalOutcome(completed_at=OLD),
            event=make_event("job_cancelled", stale=True),
        )

    terminal = await sqlite_store.terminalize(
        second.token,
        CancelledTerminalOutcome(completed_at=OLD),
        event=make_event("job_cancelled"),
    )
    page = await sqlite_store.read_events(created.handle.job_id, after_sequence=0, limit=100)
    assert terminal.state == "cancelled"
    assert [event.payload.get("stale") for event in page.events] == [None, None, None, None]


async def test_mark_input_required_updates_snapshot_and_event_atomically(
    sqlite_store: SQLiteJobStore,
):
    """A failed semantic-event insert rolls back the input rows and public state transition."""
    created = await sqlite_store.create_job(make_create_job_command())
    claimed = await sqlite_store.claim_next("worker-a", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await sqlite_store.mark_running(claimed.token, (), event=make_event("job_started"))
    pending = make_pending_permission(job_id=created.handle.job_id)
    before = await sqlite_store.read_events(created.handle.job_id, after_sequence=0, limit=100)

    def reject_input_required_event(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TEMP TRIGGER reject_input_required_event
            BEFORE INSERT ON job_events
            WHEN NEW.event_type = 'input_required'
            BEGIN
              SELECT RAISE(ABORT, 'input event rejected');
            END
            """
        )

    await sqlite_store._worker._call(reject_input_required_event)
    with pytest.raises(sqlite3.IntegrityError, match="input event rejected"):
        await sqlite_store.mark_input_required(
            claimed.token,
            (pending,),
            event=BackendEvent(type="input_required", occurred_at=OLD),
        )

    job = await sqlite_store.get_job(created.handle.job_id)
    after = await sqlite_store.read_events(created.handle.job_id, after_sequence=0, limit=100)
    assert job is not None and job.state == "running"
    assert await sqlite_store.get_pending_inputs(created.handle.job_id) == ()
    assert after.events == before.events


async def test_prune_retains_one_session_reference_recorded_by_multiple_attempts(
    sqlite_store: SQLiteJobStore,
):
    """Pruning collapses attempt rows while retaining the session's provider checkpoint."""
    created = await sqlite_store.create_job(make_create_job_command())
    reference = ProviderReference(kind="thread", value="thread-retained")
    first = await sqlite_store.claim_next("worker-a", LEASE_UNTIL, event=make_event("progress"))
    assert first is not None
    await sqlite_store.record_provider_reference(first.token, reference)
    assert await sqlite_store.renew_lease(first.token, OLD) is True
    second = await sqlite_store.claim_next(
        "worker-b", LEASE_UNTIL, event=make_event("reconciliation")
    )
    assert second is not None
    await sqlite_store.record_provider_reference(second.token, reference)
    await sqlite_store.terminalize(
        second.token,
        CancelledTerminalOutcome(completed_at=OLD),
        event=make_event("job_cancelled"),
    )

    result = await sqlite_store.prune(PrunePolicy(terminal_job_before=NOW), now=NOW)

    assert result.terminal_jobs_deleted == 1
    assert await sqlite_store.get_job(created.handle.job_id) is None
    assert await sqlite_store.get_provider_references(session_id="session-test") == (reference,)
