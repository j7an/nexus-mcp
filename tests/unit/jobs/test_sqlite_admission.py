"""SQLite-specific admission, rollback, schema, and pagination contracts."""

import asyncio
import base64
import json
import sqlite3
from pathlib import Path

import pytest

from nexus_mcp.core import (
    BackendEvent,
    DiagnosticsOperation,
    IdempotencyConflictError,
    ProviderReference,
    TurnOperation,
    Workspace,
    WorkspaceInvalidError,
    WorkspaceSelector,
)
from nexus_mcp.jobs.sqlite_store import InvalidCursorError, SQLiteJobStore, StoreSchemaError
from nexus_mcp.jobs.store import JobAccessFilter, JobQuery
from tests.unit.jobs.test_store_contract import (
    NOW,
    make_cancel_job_command,
    make_create_job_command,
)


@pytest.fixture
async def sqlite_store(tmp_path: Path):
    """Open one migrated SQLite store for an admission test."""
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    await store.open()
    try:
        yield store
    finally:
        await store.close()


async def test_create_job_rolls_back_all_admission_state_when_job_insert_fails(
    sqlite_store: SQLiteJobStore,
    monkeypatch: pytest.MonkeyPatch,
):
    """An insert failure cannot leave the new workspace, session, key, or queued event behind."""
    command = make_create_job_command(idempotency_key="rollback-key")
    original_insert = sqlite_store._insert_job

    def raise_integrity_error(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.IntegrityError("forced job insert failure")

    monkeypatch.setattr(sqlite_store, "_insert_job", raise_integrity_error)
    with pytest.raises(sqlite3.IntegrityError, match="forced job insert failure"):
        await sqlite_store.create_job(command)

    assert await sqlite_store.get_session("session-test") is None
    with pytest.raises(WorkspaceInvalidError):
        await sqlite_store.resolve_workspace(WorkspaceSelector(workspace_id="ws-test"))

    monkeypatch.setattr(sqlite_store, "_insert_job", original_insert)
    created = await sqlite_store.create_job(command)
    assert created.created is True
    assert [
        event.sequence
        for event in (await sqlite_store.read_events(created.handle.job_id, 0, 10)).events
    ] == [1]


async def test_same_key_different_hash_conflicts(sqlite_store: SQLiteJobStore):
    """A committed SQLite idempotency scope cannot alias a different canonical request."""
    await sqlite_store.create_job(
        make_create_job_command(idempotency_key="key", operation=TurnOperation(prompt="one"))
    )

    with pytest.raises(IdempotencyConflictError):
        await sqlite_store.create_job(
            make_create_job_command(
                idempotency_key="key",
                operation=TurnOperation(prompt="two"),
            )
        )


async def test_concurrent_store_instances_resolve_one_canonical_workspace(
    sqlite_store: SQLiteJobStore,
    tmp_path: Path,
):
    """Independent process-style stores converge on one atomically inserted path identity."""
    workspace_path = tmp_path / "shared-workspace"
    workspace_path.mkdir()
    second = SQLiteJobStore(sqlite_store.path)
    await second.open()
    try:
        first_workspace, second_workspace = await asyncio.gather(
            sqlite_store.resolve_or_create_workspace(WorkspaceSelector(path=workspace_path / ".")),
            second.resolve_or_create_workspace(WorkspaceSelector(path=workspace_path)),
        )
    finally:
        await second.close()

    assert first_workspace == second_workspace

    def count_workspaces(connection: sqlite3.Connection) -> int:
        return int(connection.execute("SELECT count(*) FROM workspaces").fetchone()[0])

    assert await sqlite_store._worker._call(count_workspaces) == 1


async def test_canonical_request_hash_ignores_mapping_order(sqlite_store: SQLiteJobStore):
    """Semantically identical typed JSON replays despite different input mapping order."""
    first = make_create_job_command(
        idempotency_key="canonical-key",
        operation=TurnOperation(prompt="héllo", context={"b": 2, "a": 1}),
    )
    reordered = first.model_copy(
        update={"operation": TurnOperation(prompt="héllo", context={"a": 1, "b": 2})}
    )

    created = await sqlite_store.create_job(first)
    replayed = await sqlite_store.create_job(reordered)

    assert replayed.created is False
    assert replayed.handle == created.handle


async def test_idempotent_replay_returns_the_stable_admission_handle(
    sqlite_store: SQLiteJobStore,
):
    """A later persisted state change cannot alter the handle returned by admission replay."""
    command = make_create_job_command(idempotency_key="stable-handle")
    created = await sqlite_store.create_job(command)

    def seed_terminal_snapshot(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE jobs
            SET state = 'completed', terminal_at_ms = 1, updated_at_ms = 1
            WHERE job_id = ?
            """,
            (created.handle.job_id,),
        )

    await sqlite_store._worker._call(seed_terminal_snapshot)
    replayed = await sqlite_store.create_job(command)
    stored = await sqlite_store.get_job(created.handle.job_id)

    assert stored is not None and stored.state == "completed"
    assert replayed.created is False
    assert replayed.handle == created.handle
    assert replayed.handle.state == "queued"


async def test_source_checkpoint_membership_is_checked_before_admission_mutation(
    sqlite_store: SQLiteJobStore,
):
    """A mismatched checkpoint leaves no child/key residue and an owned checkpoint can retry."""
    source = await sqlite_store.create_job(
        make_create_job_command(session_id="source-session", idempotency_key=None)
    )
    owned = ProviderReference(kind="thread", value="thread-owned")

    def seed_source_reference(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO provider_references (
              provider_reference_id, backend_id, kind, value,
              session_id, job_id, attempt_number, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
            """,
            ("ref-owned", "codex", owned.kind, owned.value, "source-session", 1),
        )

    await sqlite_store._worker._call(seed_source_reference)
    await sqlite_store.request_cancel(make_cancel_job_command(source.handle.job_id))
    invalid = make_create_job_command(
        session_id="child-session",
        parent_session_id="source-session",
        source_checkpoint=(ProviderReference(kind="thread", value="thread-other"),),
        idempotency_key="child-key",
    )

    with pytest.raises(ValueError, match="source checkpoint"):
        await sqlite_store.create_job(invalid)
    assert await sqlite_store.get_session("child-session") is None

    valid = await sqlite_store.create_job(
        invalid.model_copy(update={"source_checkpoint": (owned,)})
    )
    assert valid.created is True
    assert valid.handle.job_id != source.handle.job_id
    assert await sqlite_store.get_provider_references(job_id=valid.handle.job_id) == (owned,)


async def test_queued_event_provider_reference_is_relational_and_reuses_checkpoint(
    sqlite_store: SQLiteJobStore,
):
    """Queued-event references use one job-scoped row while raw provider event ids stay metadata."""
    source = await sqlite_store.create_job(
        make_create_job_command(session_id="event-source", idempotency_key=None)
    )
    reference = ProviderReference(kind="thread", value="thread-event")

    def seed_source_reference(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            INSERT INTO provider_references (
              provider_reference_id, backend_id, kind, value,
              session_id, job_id, attempt_number, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
            """,
            ("ref-event-source", "codex", reference.kind, reference.value, "event-source", 1),
        )

    await sqlite_store._worker._call(seed_source_reference)
    await sqlite_store.request_cancel(make_cancel_job_command(source.handle.job_id))
    queued_event = BackendEvent(
        type="job_queued",
        payload={"status": "queued"},
        occurred_at=NOW,
        provider_event_type="thread.created",
        provider_reference=reference,
    )
    created = await sqlite_store.create_job(
        make_create_job_command(
            session_id="event-child",
            parent_session_id="event-source",
            source_checkpoint=(reference,),
            queued_event=queued_event,
        )
    )

    def inspect_and_seed_raw_metadata(
        connection: sqlite3.Connection,
    ) -> tuple[str | None, str | None, int]:
        row = connection.execute(
            """
            SELECT e.provider_event_id, e.provider_reference_id, count(r.provider_reference_id)
            FROM job_events AS e
            LEFT JOIN provider_references AS r ON r.job_id = e.job_id
            WHERE e.job_id = ?
            GROUP BY e.job_id, e.sequence
            """,
            (created.handle.job_id,),
        ).fetchone()
        assert row is not None
        connection.execute(
            "UPDATE job_events SET provider_event_id = ? WHERE job_id = ?",
            ("provider-event-raw", created.handle.job_id),
        )
        return row

    provider_event_id, provider_reference_id, reference_count = await sqlite_store._worker._call(
        inspect_and_seed_raw_metadata
    )
    event = (await sqlite_store.read_events(created.handle.job_id, 0, 10)).events[0]

    assert provider_event_id is None
    assert provider_reference_id is not None
    assert reference_count == 1
    assert event.provider_event_type == "thread.created"
    assert event.provider_reference == reference


async def test_event_only_provider_reference_does_not_become_a_source_checkpoint(
    sqlite_store: SQLiteJobStore,
):
    """A queued-event reference cannot change the immutable source-checkpoint snapshot."""
    reference = ProviderReference(kind="request", value="request-event")
    created = await sqlite_store.create_job(
        make_create_job_command(
            queued_event=BackendEvent(
                type="job_queued",
                occurred_at=NOW,
                provider_reference=reference,
            )
        )
    )

    job = await sqlite_store.get_job(created.handle.job_id)
    event = (await sqlite_store.read_events(created.handle.job_id, 0, 10)).events[0]

    assert job is not None and job.source_checkpoint == ()
    assert event.provider_reference == reference


@pytest.mark.parametrize(
    "version_column",
    ["operation_schema_version", "requested_config_schema_version"],
)
async def test_job_reads_reject_unknown_typed_json_schema_versions(
    sqlite_store: SQLiteJobStore,
    version_column: str,
):
    """A future operation or request-config schema never escapes as an untyped dictionary."""
    created = await sqlite_store.create_job(make_create_job_command())

    def install_future_version(connection: sqlite3.Connection) -> None:
        connection.execute(
            f"UPDATE jobs SET {version_column} = 2 WHERE job_id = ?",
            (created.handle.job_id,),
        )

    await sqlite_store._worker._call(install_future_version)
    with pytest.raises(StoreSchemaError, match=version_column):
        await sqlite_store.get_job(created.handle.job_id)


async def test_event_reads_reject_unknown_payload_schema_version(sqlite_store: SQLiteJobStore):
    """A future event payload schema fails closed at the SQLite model boundary."""
    created = await sqlite_store.create_job(make_create_job_command())

    def install_future_version(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE job_events SET payload_schema_version = 2 WHERE job_id = ?",
            (created.handle.job_id,),
        )

    await sqlite_store._worker._call(install_future_version)
    with pytest.raises(StoreSchemaError, match="payload_schema_version"):
        await sqlite_store.read_events(created.handle.job_id, 0, 10)


@pytest.mark.parametrize(
    "cursor",
    [
        "not-base64!",
        base64.urlsafe_b64encode(
            json.dumps({"v": 2, "created_at_ms": 1, "job_id": "job"}, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        .decode("ascii")
        .rstrip("="),
        base64.urlsafe_b64encode(
            json.dumps(
                {"v": True, "created_at_ms": 1, "job_id": "job"}, separators=(",", ":")
            ).encode("utf-8")
        )
        .decode("ascii")
        .rstrip("="),
        "eyJ2IjoxLCJjcmVhdGVkX2F0X21zIjoxLCJqb2JfaWQiOiI+In0",
        base64.urlsafe_b64encode(
            json.dumps({"v": 1, "created_at_ms": 1, "job_id": "job"}, separators=(",", ":")).encode(
                "utf-8"
            )
        ).decode("ascii"),
        " eyJ2IjoxLCJjcmVhdGVkX2F0X21zIjoxLCJqb2JfaWQiOiJqb2IifQ",
    ],
    ids=[
        "malformed",
        "future-version",
        "boolean-version",
        "standard-base64",
        "padded-base64",
        "whitespace",
    ],
)
async def test_list_jobs_rejects_malformed_or_future_cursors(
    sqlite_store: SQLiteJobStore,
    cursor: str,
):
    """Opaque pagination cursors are structurally validated and explicitly versioned."""
    await sqlite_store.create_job(
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
        )
    )
    query = JobQuery(
        workspace_id="ws-test",
        access=JobAccessFilter(principal_id="local:501"),
        states={"queued"},
        limit=10,
        cursor=cursor,
    )

    with pytest.raises(InvalidCursorError):
        await sqlite_store.list_jobs(query)


async def test_list_jobs_uses_exact_created_at_and_job_id_keyset(sqlite_store: SQLiteJobStore):
    """Tied millisecond timestamps page exactly once in descending job-id order."""
    created = [
        await sqlite_store.create_job(
            make_create_job_command(
                operation=DiagnosticsOperation(),
                session_id=None,
                create_session=False,
            )
        )
        for _ in range(3)
    ]

    def tie_creation_times(connection: sqlite3.Connection) -> None:
        connection.execute("UPDATE jobs SET created_at_ms = 1, updated_at_ms = 1")

    await sqlite_store._worker._call(tie_creation_times)
    query = JobQuery(
        workspace_id="ws-test",
        access=JobAccessFilter(principal_id="local:501"),
        states={"queued"},
        limit=2,
    )
    first = await sqlite_store.list_jobs(query)
    second = await sqlite_store.list_jobs(query.model_copy(update={"cursor": first.next_cursor}))

    expected_ids = sorted((item.handle.job_id for item in created), reverse=True)
    assert [job.job_id for job in (*first.jobs, *second.jobs)] == expected_ids
    assert first.next_cursor is not None
    assert second.next_cursor is None


async def test_workspace_id_read_survives_removed_path_but_path_lookup_fails(
    sqlite_store: SQLiteJobStore,
    tmp_path: Path,
):
    """Durable identity reads do not require the admitted directory to remain on disk."""
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    workspace = Workspace(
        workspace_id="ws-removed",
        canonical_path=workspace_path,
        created_at=NOW,
        updated_at=NOW,
    )
    await sqlite_store.create_job(
        make_create_job_command(
            workspace=workspace,
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
        )
    )
    workspace_path.rmdir()

    assert (
        await sqlite_store.resolve_workspace(WorkspaceSelector(workspace_id="ws-removed"))
        == workspace
    )
    with pytest.raises(WorkspaceInvalidError):
        await sqlite_store.resolve_workspace(WorkspaceSelector(path=workspace_path))


async def test_canonical_typed_json_is_stored_as_utf8_with_sorted_keys(
    sqlite_store: SQLiteJobStore,
):
    """Typed persisted JSON has one deterministic compact UTF-8 representation."""
    created = await sqlite_store.create_job(
        make_create_job_command(
            operation=TurnOperation(prompt="héllo", context={"b": 2, "a": 1}),
        )
    )

    def read_operation_json(connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT operation_json FROM jobs WHERE job_id = ?",
            (created.handle.job_id,),
        ).fetchone()
        assert row is not None
        return str(row[0])

    operation_json = await sqlite_store._worker._call(read_operation_json)
    assert operation_json == (
        '{"context":{"a":1,"b":2},"file_refs":[],"kind":"turn","prompt":"héllo"}'
    )
    assert operation_json.encode("utf-8").decode("utf-8") == operation_json
