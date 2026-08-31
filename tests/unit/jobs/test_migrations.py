"""Versioned SQLite schema migration behavior."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from nexus_mcp.jobs.migrations import MIGRATIONS
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore

type ColumnSpec = tuple[str, str, bool, str | None, int]
type ForeignKeySpec = tuple[str, str, str, str, str]

EXPECTED_COLUMNS: dict[str, tuple[ColumnSpec, ...]] = {
    "schema_migrations": (
        ("migration_id", "TEXT", False, None, 1),
        ("checksum", "TEXT", True, None, 0),
        ("applied_at_ms", "INTEGER", True, None, 0),
    ),
    "workspaces": (
        ("workspace_id", "TEXT", False, None, 1),
        ("canonical_path", "TEXT", True, None, 0),
        ("display_name", "TEXT", False, None, 0),
        ("config_ref", "TEXT", False, None, 0),
        ("created_at_ms", "INTEGER", True, None, 0),
        ("updated_at_ms", "INTEGER", True, None, 0),
    ),
    "sessions": (
        ("session_id", "TEXT", False, None, 1),
        ("workspace_id", "TEXT", True, None, 0),
        ("backend_id", "TEXT", True, None, 0),
        ("owner_id", "TEXT", True, None, 0),
        ("access_policy", "TEXT", True, None, 0),
        ("parent_session_id", "TEXT", False, None, 0),
        ("created_at_ms", "INTEGER", True, None, 0),
        ("updated_at_ms", "INTEGER", True, None, 0),
    ),
    "jobs": (
        ("job_id", "TEXT", False, None, 1),
        ("session_id", "TEXT", False, None, 0),
        ("workspace_id", "TEXT", True, None, 0),
        ("backend_id", "TEXT", True, None, 0),
        ("owner_id", "TEXT", True, None, 0),
        ("access_policy", "TEXT", True, None, 0),
        ("operation_kind", "TEXT", True, None, 0),
        ("operation_json", "TEXT", True, None, 0),
        ("operation_schema_version", "INTEGER", True, None, 0),
        ("request_hash", "TEXT", True, None, 0),
        ("requested_config_json", "TEXT", True, None, 0),
        ("requested_config_schema_version", "INTEGER", True, None, 0),
        ("resolved_config_json", "TEXT", False, None, 0),
        ("resolved_config_schema_version", "INTEGER", False, None, 0),
        ("state", "TEXT", True, None, 0),
        ("phase", "TEXT", False, None, 0),
        ("cancel_requested_at_ms", "INTEGER", False, None, 0),
        ("retry_at_ms", "INTEGER", False, None, 0),
        ("lease_owner", "TEXT", False, None, 0),
        ("lease_generation", "INTEGER", True, "0", 0),
        ("lease_expires_at_ms", "INTEGER", False, None, 0),
        ("created_at_ms", "INTEGER", True, None, 0),
        ("updated_at_ms", "INTEGER", True, None, 0),
        ("terminal_at_ms", "INTEGER", False, None, 0),
    ),
    "job_attempts": (
        ("job_id", "TEXT", True, None, 1),
        ("attempt_number", "INTEGER", True, None, 2),
        ("phase", "TEXT", True, None, 0),
        ("owner_id", "TEXT", True, None, 0),
        ("lease_generation", "INTEGER", True, None, 0),
        ("started_at_ms", "INTEGER", True, None, 0),
        ("ended_at_ms", "INTEGER", False, None, 0),
        ("error_json", "TEXT", False, None, 0),
        ("error_schema_version", "INTEGER", False, None, 0),
    ),
    "provider_references": (
        ("provider_reference_id", "TEXT", False, None, 1),
        ("backend_id", "TEXT", True, None, 0),
        ("kind", "TEXT", True, None, 0),
        ("value", "TEXT", True, None, 0),
        ("session_id", "TEXT", False, None, 0),
        ("job_id", "TEXT", False, None, 0),
        ("attempt_number", "INTEGER", False, None, 0),
        ("created_at_ms", "INTEGER", True, None, 0),
    ),
    "pending_inputs": (
        ("input_id", "TEXT", False, None, 1),
        ("job_id", "TEXT", True, None, 0),
        ("kind", "TEXT", True, None, 0),
        ("request_json", "TEXT", True, None, 0),
        ("request_schema_version", "INTEGER", True, None, 0),
        ("response_json", "TEXT", False, None, 0),
        ("response_schema_version", "INTEGER", False, None, 0),
        ("status", "TEXT", True, None, 0),
        ("provider_reference_id", "TEXT", False, None, 0),
        ("created_at_ms", "INTEGER", True, None, 0),
        ("resolved_at_ms", "INTEGER", False, None, 0),
    ),
    "job_events": (
        ("job_id", "TEXT", True, None, 1),
        ("sequence", "INTEGER", True, None, 2),
        ("event_type", "TEXT", True, None, 0),
        ("payload_json", "TEXT", True, None, 0),
        ("payload_schema_version", "INTEGER", True, None, 0),
        ("attempt_number", "INTEGER", False, None, 0),
        ("created_at_ms", "INTEGER", True, None, 0),
        ("provider_event_type", "TEXT", False, None, 0),
        ("provider_event_id", "TEXT", False, None, 0),
    ),
    "job_results": (
        ("job_id", "TEXT", False, None, 1),
        ("outcome_kind", "TEXT", True, None, 0),
        ("payload_json", "TEXT", False, None, 0),
        ("payload_schema_version", "INTEGER", False, None, 0),
        ("error_json", "TEXT", False, None, 0),
        ("error_schema_version", "INTEGER", False, None, 0),
        ("created_at_ms", "INTEGER", True, None, 0),
    ),
    "idempotency_keys": (
        ("idempotency_id", "TEXT", False, None, 1),
        ("principal_id", "TEXT", True, None, 0),
        ("workspace_id", "TEXT", True, None, 0),
        ("command_family", "TEXT", True, None, 0),
        ("idempotency_key", "TEXT", True, None, 0),
        ("source_session_id", "TEXT", False, None, 0),
        ("request_hash", "TEXT", True, None, 0),
        ("job_id", "TEXT", True, None, 0),
        ("created_at_ms", "INTEGER", True, None, 0),
    ),
    "runtime_leases": (
        ("runtime_key", "TEXT", False, None, 1),
        ("owner_id", "TEXT", True, None, 0),
        ("lease_generation", "INTEGER", True, None, 0),
        ("endpoint", "TEXT", False, None, 0),
        ("lease_expires_at_ms", "INTEGER", True, None, 0),
        ("heartbeat_at_ms", "INTEGER", True, None, 0),
    ),
}

EXPECTED_FOREIGN_KEYS: dict[str, set[ForeignKeySpec]] = {
    "schema_migrations": set(),
    "workspaces": set(),
    "sessions": {
        ("workspace_id", "workspaces", "workspace_id", "NO ACTION", "NO ACTION"),
        ("parent_session_id", "sessions", "session_id", "NO ACTION", "NO ACTION"),
    },
    "jobs": {
        ("session_id", "sessions", "session_id", "NO ACTION", "NO ACTION"),
        ("workspace_id", "workspaces", "workspace_id", "NO ACTION", "NO ACTION"),
    },
    "job_attempts": {("job_id", "jobs", "job_id", "NO ACTION", "NO ACTION")},
    "provider_references": {
        ("session_id", "sessions", "session_id", "NO ACTION", "NO ACTION"),
        ("job_id", "jobs", "job_id", "NO ACTION", "NO ACTION"),
    },
    "pending_inputs": {
        ("job_id", "jobs", "job_id", "NO ACTION", "NO ACTION"),
        (
            "provider_reference_id",
            "provider_references",
            "provider_reference_id",
            "NO ACTION",
            "NO ACTION",
        ),
    },
    "job_events": {("job_id", "jobs", "job_id", "NO ACTION", "NO ACTION")},
    "job_results": {("job_id", "jobs", "job_id", "NO ACTION", "NO ACTION")},
    "idempotency_keys": {
        ("workspace_id", "workspaces", "workspace_id", "NO ACTION", "NO ACTION"),
        ("source_session_id", "sessions", "session_id", "NO ACTION", "NO ACTION"),
        ("job_id", "jobs", "job_id", "NO ACTION", "NO ACTION"),
    },
    "runtime_leases": set(),
}

EXPECTED_INDEXES = {
    "claimable_jobs": ("jobs", False, False),
    "job_event_sequence": ("job_events", True, False),
    "jobs_by_workspace_created": ("jobs", False, False),
    "one_nonterminal_job_per_session": ("jobs", True, True),
    "provider_reference_identity": ("provider_references", True, False),
    "scoped_idempotency_key": ("idempotency_keys", True, False),
}

EXPECTED_INDEX_SQL = {
    "one_nonterminal_job_per_session": (
        "CREATE UNIQUE INDEX one_nonterminal_job_per_session ON jobs(session_id) "
        "WHERE session_id IS NOT NULL "
        "AND state IN ('queued', 'running', 'input_required')"
    ),
    "scoped_idempotency_key": (
        "CREATE UNIQUE INDEX scoped_idempotency_key ON idempotency_keys("
        "principal_id, workspace_id, command_family, idempotency_key, "
        "ifnull(source_session_id, ''))"
    ),
    "job_event_sequence": (
        "CREATE UNIQUE INDEX job_event_sequence ON job_events(job_id, sequence)"
    ),
    "provider_reference_identity": (
        "CREATE UNIQUE INDEX provider_reference_identity ON provider_references("
        "backend_id, kind, value, ifnull(session_id, ''), ifnull(job_id, ''), "
        "ifnull(attempt_number, -1))"
    ),
    "claimable_jobs": (
        "CREATE INDEX claimable_jobs "
        "ON jobs(state, retry_at_ms, lease_expires_at_ms, created_at_ms)"
    ),
    "jobs_by_workspace_created": (
        "CREATE INDEX jobs_by_workspace_created "
        "ON jobs(workspace_id, created_at_ms DESC, job_id DESC)"
    ),
}

EXPECTED_CHECKS = {
    "sessions": ("CHECK (access_policy IN ('private', 'workspace'))",),
    "jobs": (
        "CHECK (access_policy IN ('private', 'workspace'))",
        "CHECK (operation_kind IN ('turn','fork','review','diagnostics'))",
        "CHECK (state IN ('queued','running','input_required','completed','failed','cancelled'))",
    ),
    "provider_references": ("CHECK (session_id IS NOT NULL OR job_id IS NOT NULL)",),
    "pending_inputs": (
        "CHECK (kind IN ('approval','permission','question','form'))",
        "CHECK (status IN ('pending','resolved','expired'))",
    ),
    "job_results": ("CHECK (outcome_kind IN ('succeeded','failed','cancelled'))",),
}


async def create_database(path: Path) -> None:
    """Open and close a store so schema assertions can use an independent connection."""
    store = SQLiteJobStore(path)
    await store.open()
    await store.close()


def read_columns(connection: sqlite3.Connection, table: str) -> tuple[ColumnSpec, ...]:
    """Return complete column semantics in durable declaration order."""
    return tuple(
        (row[1], row[2], bool(row[3]), row[4], row[5])
        for row in connection.execute(f"PRAGMA table_info({table})")
    )


def read_foreign_keys(connection: sqlite3.Connection, table: str) -> set[ForeignKeySpec]:
    """Return foreign-key targets plus update and deletion behavior."""
    return {
        (row[3], row[2], row[4], row[5], row[6])
        for row in connection.execute(f"PRAGMA foreign_key_list({table})")
    }


def normalize_sql(sql: str) -> str:
    """Remove formatting-only differences while retaining SQL expression order."""
    return " ".join(sql.split()).replace("( ", "(").replace(" )", ")")


def insert_job(
    connection: sqlite3.Connection,
    job_id: str,
    *,
    session_id: str | None = None,
    access_policy: str = "private",
    operation_kind: str = "diagnostics",
    state: str = "queued",
) -> None:
    """Insert one minimal job for schema-constraint behavior tests."""
    connection.execute(
        """
        INSERT INTO jobs (
          job_id, session_id, workspace_id, backend_id, owner_id, access_policy,
          operation_kind, operation_json, operation_schema_version, request_hash,
          requested_config_json, requested_config_schema_version, state,
          created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job_id,
            session_id,
            "workspace",
            "codex",
            "owner",
            access_policy,
            operation_kind,
            "{}",
            1,
            "hash",
            "{}",
            1,
            state,
            1,
            1,
        ),
    )


async def test_initial_migration_creates_complete_schema(tmp_path):
    """A new database exposes the complete semantic schema and migration record."""
    database_path = tmp_path / "schema.sqlite3"
    await create_database(database_path)

    with sqlite3.connect(database_path) as connection:
        table_sql = {
            row[0]: normalize_sql(row[1])
            for row in connection.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        columns = {table: read_columns(connection, table) for table in table_sql}
        foreign_keys = {table: read_foreign_keys(connection, table) for table in table_sql}
        index_rows = {
            row[0]: (row[1], normalize_sql(row[2]))
            for row in connection.execute(
                "SELECT name, tbl_name, sql FROM sqlite_master "
                "WHERE type = 'index' AND sql IS NOT NULL"
            )
        }
        indexes = {}
        for index_name, (table, _sql) in index_rows.items():
            index_list_row = next(
                row
                for row in connection.execute(f"PRAGMA index_list({table})")
                if row[1] == index_name
            )
            indexes[index_name] = (table, bool(index_list_row[2]), bool(index_list_row[4]))
        migration_rows = connection.execute(
            "SELECT migration_id, checksum FROM schema_migrations"
        ).fetchall()

    assert columns == EXPECTED_COLUMNS
    assert foreign_keys == EXPECTED_FOREIGN_KEYS
    assert indexes == EXPECTED_INDEXES
    assert {name: sql for name, (_table, sql) in index_rows.items()} == EXPECTED_INDEX_SQL
    assert migration_rows == [(MIGRATIONS[0].migration_id, MIGRATIONS[0].checksum)]
    assert sum(sql.count("CHECK (") for sql in table_sql.values()) == sum(
        len(checks) for checks in EXPECTED_CHECKS.values()
    )
    for table, checks in EXPECTED_CHECKS.items():
        assert all(check in table_sql[table] for check in checks)


async def test_required_check_constraints_reject_invalid_values(tmp_path):
    """Every declared value-domain and reference-shape check rejects an invalid row."""
    database_path = tmp_path / "checks.sqlite3"
    await create_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO workspaces VALUES (?, ?, NULL, NULL, ?, ?)",
            ("workspace", "/tmp/workspace", 1, 1),
        )

        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
                ("bad-session", "workspace", "codex", "owner", "public", 1, 1),
            )

        for job_id, overrides in (
            ("bad-access", {"access_policy": "public"}),
            ("bad-operation", {"operation_kind": "unknown"}),
            ("bad-state", {"state": "paused"}),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
                insert_job(connection, job_id, **overrides)

        insert_job(connection, "valid-job")

        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            connection.execute(
                "INSERT INTO provider_references VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?)",
                ("bad-reference", "codex", "thread", "thread-1", 1),
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            connection.execute(
                "INSERT INTO pending_inputs VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, NULL)",
                ("bad-kind", "valid-job", "message", "{}", 1, "pending", 1),
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            connection.execute(
                "INSERT INTO pending_inputs VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, NULL, ?, NULL)",
                ("bad-status", "valid-job", "question", "{}", 1, "open", 1),
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
            connection.execute(
                "INSERT INTO job_results VALUES (?, ?, NULL, NULL, NULL, NULL, ?)",
                ("valid-job", "unknown", 1),
            )


async def test_foreign_keys_retain_referenced_execution_rows(tmp_path):
    """Restrictive foreign keys prevent deletion of referenced workspace history."""
    database_path = tmp_path / "retained.sqlite3"
    await create_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO workspaces VALUES (?, ?, NULL, NULL, ?, ?)",
            ("workspace", "/tmp/workspace", 1, 1),
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            ("session", "workspace", "codex", "owner", "private", 1, 1),
        )
        insert_job(connection, "job", session_id="session")
        connection.execute(
            "INSERT INTO job_events VALUES (?, ?, ?, ?, ?, NULL, ?, NULL, NULL)",
            ("job", 1, "job_queued", "{}", 1, 1),
        )

        for table, key_column, key in (
            ("workspaces", "workspace_id", "workspace"),
            ("sessions", "session_id", "session"),
            ("jobs", "job_id", "job"),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
                connection.execute(f"DELETE FROM {table} WHERE {key_column} = ?", (key,))
            assert connection.execute(
                f"SELECT 1 FROM {table} WHERE {key_column} = ?", (key,)
            ).fetchone() == (1,)


async def test_nonterminal_session_index_uses_exact_partial_scope(tmp_path):
    """The session uniqueness guard includes only the three nonterminal states."""
    database_path = tmp_path / "nonterminal.sqlite3"
    await create_database(database_path)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO workspaces VALUES (?, ?, NULL, NULL, ?, ?)",
            ("workspace", "/tmp/workspace", 1, 1),
        )
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            ("session", "workspace", "codex", "owner", "private", 1, 1),
        )
        insert_job(connection, "queued", session_id="session", state="queued")
        insert_job(connection, "completed", session_id="session", state="completed")

        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            insert_job(connection, "input-required", session_id="session", state="input_required")


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
