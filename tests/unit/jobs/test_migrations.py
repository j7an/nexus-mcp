"""Versioned SQLite schema migration behavior."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from nexus_mcp.jobs.migrations import MIGRATIONS
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore

EXPECTED_COLUMNS = {
    "schema_migrations": ("migration_id", "checksum", "applied_at_ms"),
    "workspaces": (
        "workspace_id",
        "canonical_path",
        "display_name",
        "config_ref",
        "created_at_ms",
        "updated_at_ms",
    ),
    "sessions": (
        "session_id",
        "workspace_id",
        "backend_id",
        "owner_id",
        "access_policy",
        "parent_session_id",
        "created_at_ms",
        "updated_at_ms",
    ),
    "jobs": (
        "job_id",
        "session_id",
        "workspace_id",
        "backend_id",
        "owner_id",
        "access_policy",
        "operation_kind",
        "operation_json",
        "operation_schema_version",
        "request_hash",
        "requested_config_json",
        "requested_config_schema_version",
        "resolved_config_json",
        "resolved_config_schema_version",
        "state",
        "phase",
        "cancel_requested_at_ms",
        "retry_at_ms",
        "lease_owner",
        "lease_generation",
        "lease_expires_at_ms",
        "created_at_ms",
        "updated_at_ms",
        "terminal_at_ms",
    ),
    "job_attempts": (
        "job_id",
        "attempt_number",
        "phase",
        "owner_id",
        "lease_generation",
        "started_at_ms",
        "ended_at_ms",
        "error_json",
        "error_schema_version",
    ),
    "provider_references": (
        "provider_reference_id",
        "backend_id",
        "kind",
        "value",
        "session_id",
        "job_id",
        "attempt_number",
        "created_at_ms",
    ),
    "pending_inputs": (
        "input_id",
        "job_id",
        "kind",
        "request_json",
        "request_schema_version",
        "response_json",
        "response_schema_version",
        "status",
        "provider_reference_id",
        "created_at_ms",
        "resolved_at_ms",
    ),
    "job_events": (
        "job_id",
        "sequence",
        "event_type",
        "payload_json",
        "payload_schema_version",
        "attempt_number",
        "created_at_ms",
        "provider_event_type",
        "provider_event_id",
    ),
    "job_results": (
        "job_id",
        "outcome_kind",
        "payload_json",
        "payload_schema_version",
        "error_json",
        "error_schema_version",
        "created_at_ms",
    ),
    "idempotency_keys": (
        "idempotency_id",
        "principal_id",
        "workspace_id",
        "command_family",
        "idempotency_key",
        "source_session_id",
        "request_hash",
        "job_id",
        "created_at_ms",
    ),
    "runtime_leases": (
        "runtime_key",
        "owner_id",
        "lease_generation",
        "endpoint",
        "lease_expires_at_ms",
        "heartbeat_at_ms",
    ),
}

EXPECTED_INDEXES = {
    "claimable_jobs",
    "job_event_sequence",
    "jobs_by_workspace_created",
    "one_nonterminal_job_per_session",
    "provider_reference_identity",
    "scoped_idempotency_key",
}


async def create_database(path: Path) -> None:
    """Open and close a store so schema assertions can use an independent connection."""
    store = SQLiteJobStore(path)
    await store.open()
    await store.close()


def read_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    """Return columns in their durable declaration order."""
    return tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))


async def test_initial_migration_creates_complete_schema(tmp_path):
    """A new database exposes every required table, column, index, and migration record."""
    database_path = tmp_path / "schema.sqlite3"
    await create_database(database_path)

    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {table: read_columns(connection, table) for table in tables}
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"
            )
        }
        migration_rows = connection.execute(
            "SELECT migration_id, checksum FROM schema_migrations"
        ).fetchall()
        source_session_foreign_key = connection.execute(
            "PRAGMA foreign_key_list(idempotency_keys)"
        ).fetchall()

    assert columns == EXPECTED_COLUMNS
    assert indexes == EXPECTED_INDEXES
    assert migration_rows == [(MIGRATIONS[0].migration_id, MIGRATIONS[0].checksum)]
    assert any(
        row[2] == "sessions" and row[3] == "source_session_id" and row[4] == "session_id"
        for row in source_session_foreign_key
    )


async def test_idempotency_scope_includes_nullable_source_session(tmp_path):
    """Identical request keys are independent by source session, including one null scope."""
    database_path = tmp_path / "idempotency.sqlite3"
    await create_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO workspaces VALUES (?, ?, NULL, NULL, ?, ?)",
            ("workspace", "/tmp/workspace", 1, 1),
        )
        connection.executemany(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                ("source-a", "workspace", "codex", "owner", "private", 1, 1),
                ("source-b", "workspace", "codex", "owner", "private", 1, 1),
            ),
        )
        connection.execute(
            """
            INSERT INTO jobs (
              job_id, workspace_id, backend_id, owner_id, access_policy, operation_kind,
              operation_json, operation_schema_version, request_hash, requested_config_json,
              requested_config_schema_version, state, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job",
                "workspace",
                "codex",
                "owner",
                "private",
                "diagnostics",
                "{}",
                1,
                "hash",
                "{}",
                1,
                "queued",
                1,
                1,
            ),
        )
        for idempotency_id, source_session_id in (
            ("null-scope", None),
            ("source-a-scope", "source-a"),
            ("source-b-scope", "source-b"),
        ):
            connection.execute(
                """
                INSERT INTO idempotency_keys VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_id,
                    "owner",
                    "workspace",
                    "submit",
                    "same-key",
                    source_session_id,
                    "hash",
                    "job",
                    1,
                ),
            )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO idempotency_keys VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "duplicate-null",
                    "owner",
                    "workspace",
                    "submit",
                    "same-key",
                    None,
                    "hash",
                    "job",
                    1,
                ),
            )


async def test_repeat_open_does_not_reapply_migration(tmp_path):
    """Opening one store twice leaves the original migration record untouched."""
    database_path = tmp_path / "repeat.sqlite3"
    store = SQLiteJobStore(database_path)

    await store.open()
    with sqlite3.connect(database_path) as connection:
        before = connection.execute("SELECT * FROM schema_migrations").fetchall()
    await store.open()
    await store.close()
    with sqlite3.connect(database_path) as connection:
        after = connection.execute("SELECT * FROM schema_migrations").fetchall()

    assert after == before
    assert len(after) == 1


async def test_checksum_mismatch_fails_closed(tmp_path):
    """Changed migration contents cannot silently reinterpret an existing schema."""
    database_path = tmp_path / "checksum.sqlite3"
    await create_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE schema_migrations SET checksum = 'tampered'")

    store = SQLiteJobStore(database_path)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        await store.open()
    await store.close()


async def test_newer_schema_fails_closed(tmp_path):
    """Code must not open a database containing an unknown future migration."""
    database_path = tmp_path / "newer.sqlite3"
    await create_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("INSERT INTO schema_migrations VALUES ('v9999_future', 'future', 9999)")

    store = SQLiteJobStore(database_path)
    with pytest.raises(RuntimeError, match="newer"):
        await store.open()
    await store.close()


async def test_concurrent_open_converges_on_one_schema(tmp_path):
    """Two independent worker connections serialize initial migration safely."""
    database_path = tmp_path / "concurrent.sqlite3"
    first = SQLiteJobStore(database_path)
    second = SQLiteJobStore(database_path)

    await asyncio.gather(first.open(), second.open())
    try:
        first_connection_id = await first._worker._call(id)
        second_connection_id = await second._worker._call(id)
        with sqlite3.connect(database_path) as connection:
            rows = connection.execute(
                "SELECT migration_id, checksum FROM schema_migrations"
            ).fetchall()
    finally:
        await asyncio.gather(first.close(), second.close())

    assert first_connection_id != second_connection_id
    assert rows == [(MIGRATIONS[0].migration_id, MIGRATIONS[0].checksum)]
