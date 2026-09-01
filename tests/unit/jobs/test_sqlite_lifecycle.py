"""SQLite-specific lifecycle fencing and transaction regressions."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nexus_mcp.core import BackendEvent, ProviderReference, StaleLeaseError
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore, StoreSchemaError
from nexus_mcp.jobs.store import (
    CancelledTerminalOutcome,
    FailedTerminalOutcome,
    PrunePolicy,
    SucceededTerminalOutcome,
    TerminalOutcome,
)
from tests.fixtures import make_job_error, make_pending_permission, make_turn_result
from tests.unit.jobs._store_contract_support import make_create_job_command, make_event

OLD = datetime(2025, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 8, 30, 20, 0, tzinfo=UTC)
LEASE_UNTIL = datetime(2099, 1, 1, tzinfo=UTC)


def _execute_sql(connection: sqlite3.Connection, sql: str, parameters: tuple[object, ...]) -> None:
    connection.execute(sql, parameters)


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


async def _terminalize_for_result(
    sqlite_store: SQLiteJobStore,
    outcome: TerminalOutcome,
) -> str:
    created = await sqlite_store.create_job(make_create_job_command())
    claimed = await sqlite_store.claim_next("worker-a", LEASE_UNTIL, event=make_event("progress"))
    assert claimed is not None
    await sqlite_store.mark_running(claimed.token, (), event=make_event("job_started"))
    event_type = {
        "succeeded": "job_completed",
        "failed": "job_failed",
        "cancelled": "job_cancelled",
    }[outcome.kind]
    await sqlite_store.terminalize(
        claimed.token,
        outcome,
        event=make_event(event_type),
    )
    return created.handle.job_id


@pytest.mark.parametrize(
    "outcome",
    [
        SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
        FailedTerminalOutcome(error=make_job_error(), completed_at=NOW),
        CancelledTerminalOutcome(completed_at=NOW),
    ],
    ids=["succeeded", "failed", "cancelled"],
)
async def test_job_result_rejects_missing_terminal_row(
    sqlite_store: SQLiteJobStore,
    outcome: TerminalOutcome,
):
    """Every terminal snapshot requires its matching normalized result row."""
    job_id = await _terminalize_for_result(sqlite_store, outcome)
    await sqlite_store._worker._call(
        lambda connection: _execute_sql(
            connection,
            "DELETE FROM job_results WHERE job_id = ?",
            (job_id,),
        )
    )

    with pytest.raises(StoreSchemaError, match="result"):
        await sqlite_store.get_job_result(job_id)


async def test_job_result_rejects_a_row_for_nonterminal_job(sqlite_store: SQLiteJobStore):
    """A queued job cannot expose a fabricated terminal result row."""
    created = await sqlite_store.create_job(make_create_job_command())

    def insert_result(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO job_results (
              job_id, outcome_kind, payload_json, payload_schema_version,
              error_json, error_schema_version, created_at_ms
            ) VALUES (?, 'cancelled', NULL, NULL, NULL, NULL, ?)
            """,
            (created.handle.job_id, int(NOW.timestamp() * 1000)),
        )

    await sqlite_store._worker._call(insert_result)
    with pytest.raises(StoreSchemaError, match="nonterminal"):
        await sqlite_store.get_job_result(created.handle.job_id)


@pytest.mark.parametrize(
    ("outcome", "mutation"),
    [
        (
            SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
            "UPDATE job_results SET payload_json = NULL WHERE job_id = ?",
        ),
        (
            SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
            "UPDATE job_results SET payload_schema_version = NULL WHERE job_id = ?",
        ),
        (
            SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
            "UPDATE job_results SET error_json = '{}', error_schema_version = 1 WHERE job_id = ?",
        ),
        (
            SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
            "UPDATE job_results SET payload_schema_version = 2 WHERE job_id = ?",
        ),
        (
            SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
            "UPDATE jobs SET state = 'failed' WHERE job_id = ?",
        ),
        (
            FailedTerminalOutcome(error=make_job_error(), completed_at=NOW),
            "UPDATE job_results SET error_json = NULL WHERE job_id = ?",
        ),
        (
            FailedTerminalOutcome(error=make_job_error(), completed_at=NOW),
            "UPDATE job_results SET error_schema_version = NULL WHERE job_id = ?",
        ),
        (
            FailedTerminalOutcome(error=make_job_error(), completed_at=NOW),
            (
                "UPDATE job_results SET payload_json = '{}', "
                "payload_schema_version = 1 WHERE job_id = ?"
            ),
        ),
        (
            FailedTerminalOutcome(error=make_job_error(), completed_at=NOW),
            "UPDATE job_results SET error_schema_version = 2 WHERE job_id = ?",
        ),
        (
            FailedTerminalOutcome(error=make_job_error(), completed_at=NOW),
            "UPDATE jobs SET state = 'completed' WHERE job_id = ?",
        ),
        (
            CancelledTerminalOutcome(completed_at=NOW),
            "UPDATE job_results SET payload_schema_version = 1 WHERE job_id = ?",
        ),
        (
            CancelledTerminalOutcome(completed_at=NOW),
            "UPDATE jobs SET state = 'completed' WHERE job_id = ?",
        ),
    ],
    ids=[
        "succeeded-missing-payload",
        "succeeded-missing-version",
        "succeeded-extra-error",
        "succeeded-future-version",
        "succeeded-state-mismatch",
        "failed-missing-error",
        "failed-missing-version",
        "failed-extra-payload",
        "failed-future-version",
        "failed-state-mismatch",
        "cancelled-extra-version",
        "cancelled-state-mismatch",
    ],
)
async def test_job_result_rejects_corrupt_shape_or_state(
    sqlite_store: SQLiteJobStore,
    outcome: TerminalOutcome,
    mutation: str,
):
    """Result variants reject missing, extra, future-version, and state-mismatched fields."""
    job_id = await _terminalize_for_result(sqlite_store, outcome)
    await sqlite_store._worker._call(
        lambda connection: _execute_sql(connection, mutation, (job_id,))
    )

    with pytest.raises(StoreSchemaError):
        await sqlite_store.get_job_result(job_id)


async def test_job_result_rejects_unknown_outcome_kind(sqlite_store: SQLiteJobStore):
    """Reader validation remains closed even if a damaged database bypasses its CHECK constraint."""
    job_id = await _terminalize_for_result(
        sqlite_store,
        CancelledTerminalOutcome(completed_at=NOW),
    )

    def corrupt_outcome(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        try:
            connection.execute(
                "UPDATE job_results SET outcome_kind = 'unknown' WHERE job_id = ?",
                (job_id,),
            )
        finally:
            connection.execute("PRAGMA ignore_check_constraints = OFF")

    await sqlite_store._worker._call(corrupt_outcome)
    with pytest.raises(StoreSchemaError, match="outcome"):
        await sqlite_store.get_job_result(job_id)
