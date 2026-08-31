"""Dedicated-thread SQLite lifecycle and schema foundation for durable jobs."""

import asyncio
import base64
import binascii
import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from nexus_mcp.core import (
    AgentJob,
    AgentOperation,
    AgentSession,
    BackendEvent,
    IdempotencyConflictError,
    JobEvent,
    JobHandle,
    JobNotFoundError,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    SessionBusyError,
    SessionNotFoundError,
    Workspace,
    WorkspaceInvalidError,
    WorkspaceSelector,
    new_id,
)
from nexus_mcp.jobs.migrations import MIGRATIONS, Migration
from nexus_mcp.jobs.paths import default_database_path
from nexus_mcp.jobs.store import (
    CreateJobCommand,
    CreateJobResult,
    EventPage,
    JobQuery,
    StoredJobPage,
)

__all__ = ["InvalidCursorError", "SQLiteJobStore", "StoreSchemaError"]

_ResultT = TypeVar("_ResultT")
_JSON_SCHEMA_VERSION = 1
_CURSOR_VERSION = 1
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_OPERATION_ADAPTER: TypeAdapter[AgentOperation] = TypeAdapter(AgentOperation)
_JOB_SELECT = """
SELECT j.*,
       (
         SELECT i.idempotency_key
         FROM idempotency_keys AS i
         WHERE i.job_id = j.job_id
         LIMIT 1
       ) AS idempotency_key
FROM jobs AS j
"""

_CONNECTION_PRAGMAS = (
    "PRAGMA journal_mode = WAL;",
    "PRAGMA foreign_keys = ON;",
    "PRAGMA synchronous = FULL;",
    "PRAGMA busy_timeout = 5000;",
)


class StoreSchemaError(RuntimeError):
    """Raised when persisted typed JSON uses an unsupported schema version."""


class InvalidCursorError(ValueError):
    """Raised when a stored-job keyset cursor is malformed or unsupported."""


class _SQLiteWorker:
    """Own one SQLite connection and use it only on one dedicated thread."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nexus-sqlite")
        self._connection: sqlite3.Connection | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._closed = False

    async def open(self) -> None:
        """Create and configure the connection inside the dedicated worker."""
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("SQLite worker is closed")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._open_in_worker)

    async def _call(self, operation: Callable[[sqlite3.Connection], _ResultT]) -> _ResultT:
        """Run one trusted internal connection operation on the dedicated worker thread.

        Direct connection and cursor results are rejected. Internal operations are trusted
        not to smuggle SQLite handles through closures or wrapper containers.
        """
        if self._closed:
            raise RuntimeError("SQLite worker is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._invoke_in_worker, operation)

    async def close(self) -> None:
        """Close the connection and executor once; repeated calls are safe."""
        async with self._lifecycle_lock:
            if self._closed:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._close_in_worker)
            self._closed = True
            self._executor.shutdown(wait=True)

    def _open_in_worker(self) -> None:
        if self._connection is not None:
            return
        connection = sqlite3.connect(
            self._path,
            check_same_thread=True,
            isolation_level=None,
        )
        try:
            for pragma in _CONNECTION_PRAGMAS:
                connection.execute(pragma)
        except BaseException:
            connection.close()
            raise
        self._connection = connection

    def _invoke_in_worker(self, operation: Callable[[sqlite3.Connection], _ResultT]) -> _ResultT:
        connection = self._connection
        if connection is None:
            raise RuntimeError("SQLite worker is not open")
        result = operation(connection)
        if isinstance(result, sqlite3.Connection):
            raise RuntimeError("SQLite connection cannot escape its worker")
        if isinstance(result, sqlite3.Cursor):
            raise RuntimeError("SQLite cursor cannot escape its worker")
        return result

    def _close_in_worker(self) -> None:
        if self._connection is None:
            return
        self._connection.close()
        self._connection = None


class SQLiteJobStore:
    """Dedicated-thread SQLite admission and query store with lifecycle added separately."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_database_path()
        self._worker = _SQLiteWorker(self.path)
        self._lifecycle_lock = asyncio.Lock()
        self._opened = False

    async def open(self) -> None:
        """Create the private database, configure SQLite, and migrate forward."""
        async with self._lifecycle_lock:
            if self._opened:
                return
            self._prepare_parent_directory()
            try:
                await self._worker.open()
                self._secure_database_files()
                await self._worker._call(_apply_migrations)
                self._secure_database_files()
            except BaseException:
                await self._worker.close()
                raise
            self._opened = True

    async def close(self) -> None:
        """Close this store once; repeated calls are safe."""
        async with self._lifecycle_lock:
            await self._worker.close()
            self._opened = False

    async def resolve_workspace(self, selector: WorkspaceSelector) -> Workspace:
        """Resolve a durable workspace id or a currently valid canonical path."""
        if selector.workspace_id is not None:
            identity = selector.workspace_id
            column = "workspace_id"
        else:
            assert selector.path is not None
            identity = _resolve_workspace_path(selector.path)
            column = "canonical_path"
        workspace = await self._worker._call(
            lambda connection: _read_workspace(connection, column, identity)
        )
        if workspace is None:
            raise WorkspaceInvalidError(identity, "workspace is not registered")
        return workspace

    async def create_job(self, command: CreateJobCommand) -> CreateJobResult:
        """Persist one admitted job atomically."""
        return await self._worker._call(
            lambda connection: self._create_job_transaction(connection, command)
        )

    async def get_session(self, session_id: str) -> AgentSession | None:
        """Return one durable session and its provider references."""
        return await self._worker._call(lambda connection: _read_session(connection, session_id))

    async def get_job(self, job_id: str) -> AgentJob | None:
        """Return one durable job reconstructed from versioned typed columns."""
        return await self._worker._call(lambda connection: _read_job(connection, job_id))

    async def get_provider_references(
        self, *, session_id: str | None = None, job_id: str | None = None
    ) -> tuple[ProviderReference, ...]:
        """Return references scoped to exactly one durable session or job."""
        if (session_id is None) == (job_id is None):
            raise ValueError("exactly one of session_id or job_id is required")
        column, identity = (
            ("session_id", session_id) if session_id is not None else ("job_id", job_id)
        )
        assert identity is not None
        return await self._worker._call(
            lambda connection: _read_provider_references(connection, column, identity)
        )

    async def list_jobs(self, query: JobQuery) -> StoredJobPage:
        """Return one authorized descending keyset page of typed job snapshots."""
        cursor = None if query.cursor is None else _decode_cursor(query.cursor)
        return await self._worker._call(lambda connection: _list_jobs(connection, query, cursor))

    async def read_events(self, job_id: str, after_sequence: int, limit: int) -> EventPage:
        """Read committed admission events after one bounded job-local sequence."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if not 1 <= limit <= 1000:
            raise ValueError("event page limit must be from 1 through 1000")
        return await self._worker._call(
            lambda connection: _read_events(connection, job_id, after_sequence, limit)
        )

    def _create_job_transaction(
        self,
        connection: sqlite3.Connection,
        command: CreateJobCommand,
    ) -> CreateJobResult:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _validate_source_checkpoint(connection, command)
            request_hash = _create_request_hash(command)
            replay = _read_idempotency_replay(connection, command, request_hash)
            if replay is not None:
                connection.execute("COMMIT")
                return replay

            _validate_workspace_identity(connection, command.workspace)
            insert_session = _validate_session_admission(connection, command)
            _raise_if_session_busy(connection, command.session_id)

            _upsert_workspace(connection, command.workspace)
            now_ms = time.time_ns() // 1_000_000
            if insert_session:
                _insert_session(connection, command, now_ms)
            job_id = new_id()
            self._insert_job(connection, job_id, command, request_hash, now_ms)
            _insert_source_checkpoint(connection, job_id, command, now_ms)
            _insert_idempotency_key(connection, job_id, command, request_hash, now_ms)
            _insert_queued_event(
                connection,
                job_id,
                command.backend_id,
                command.queued_event,
                now_ms,
            )
            connection.execute("COMMIT")
            return CreateJobResult(
                handle=JobHandle(
                    job_id=job_id,
                    session_id=command.session_id,
                    operation=command.operation,
                ),
                created=True,
            )
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _insert_job(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        command: CreateJobCommand,
        request_hash: str,
        now_ms: int,
    ) -> None:
        """Insert the immutable admitted request and its initial queued snapshot."""
        connection.execute(
            """
            INSERT INTO jobs (
              job_id, session_id, workspace_id, backend_id, owner_id, access_policy,
              operation_kind, operation_json, operation_schema_version, request_hash,
              requested_config_json, requested_config_schema_version, state,
              created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
            """,
            (
                job_id,
                command.session_id,
                command.workspace.workspace_id,
                command.backend_id,
                command.owner_id,
                command.access_policy,
                command.operation.kind,
                _canonical_json(command.operation),
                _JSON_SCHEMA_VERSION,
                request_hash,
                _canonical_json(command.requested_config),
                _JSON_SCHEMA_VERSION,
                now_ms,
                now_ms,
            ),
        )

    def _prepare_parent_directory(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == "posix":
            self.path.parent.chmod(0o700)

    def _secure_database_files(self) -> None:
        if os.name != "posix":
            return
        for path in (self.path, Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
            if path.exists():
                path.chmod(0o600)


def _resolve_workspace_path(path: Path) -> str:
    """Resolve and validate a path before any SQLite transaction begins."""
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkspaceInvalidError(str(path), "workspace path does not exist") from error
    if not resolved.is_dir():
        raise WorkspaceInvalidError(str(path), "workspace path is not a directory")
    return os.path.normcase(str(resolved))


def _canonical_workspace_path(path: Path) -> str:
    return os.path.normcase(str(path))


def _datetime_to_ms(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    delta = normalized - _EPOCH
    return delta.days * 86_400_000 + delta.seconds * 1_000 + delta.microseconds // 1_000


def _ms_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000, UTC)


def _canonical_json(value: BaseModel | object) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _create_request_hash(command: CreateJobCommand) -> str:
    payload = command.model_dump(
        mode="json",
        exclude={"idempotency_key", "queued_event"},
    )
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _fetch_one_mapping(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...] = (),
) -> dict[str, Any] | None:
    cursor = connection.execute(sql, parameters)
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(zip((column[0] for column in cursor.description), row, strict=True))


def _fetch_all_mappings(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[object, ...] = (),
) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, parameters)
    names = tuple(column[0] for column in cursor.description)
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _read_workspace(
    connection: sqlite3.Connection,
    column: str,
    identity: str,
) -> Workspace | None:
    statements = {
        "workspace_id": "SELECT * FROM workspaces WHERE workspace_id = ?",
        "canonical_path": "SELECT * FROM workspaces WHERE canonical_path = ?",
    }
    row = _fetch_one_mapping(connection, statements[column], (identity,))
    if row is None:
        return None
    return Workspace(
        workspace_id=row["workspace_id"],
        canonical_path=Path(row["canonical_path"]),
        display_name=row["display_name"],
        config_reference=row["config_ref"],
        created_at=_ms_to_datetime(row["created_at_ms"]),
        updated_at=_ms_to_datetime(row["updated_at_ms"]),
    )


def _validate_workspace_identity(
    connection: sqlite3.Connection,
    workspace: Workspace,
) -> None:
    canonical_path = _canonical_workspace_path(workspace.canonical_path)
    by_id = _fetch_one_mapping(
        connection,
        "SELECT workspace_id, canonical_path FROM workspaces WHERE workspace_id = ?",
        (workspace.workspace_id,),
    )
    if by_id is not None and by_id["canonical_path"] != canonical_path:
        raise WorkspaceInvalidError(workspace.workspace_id, "workspace path changed")
    by_path = _fetch_one_mapping(
        connection,
        "SELECT workspace_id FROM workspaces WHERE canonical_path = ?",
        (canonical_path,),
    )
    if by_path is not None and by_path["workspace_id"] != workspace.workspace_id:
        raise WorkspaceInvalidError(canonical_path, "canonical path belongs to another workspace")


def _upsert_workspace(connection: sqlite3.Connection, workspace: Workspace) -> None:
    connection.execute(
        """
        INSERT INTO workspaces (
          workspace_id, canonical_path, display_name, config_ref, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(workspace_id) DO UPDATE SET
          display_name = excluded.display_name,
          config_ref = excluded.config_ref,
          updated_at_ms = excluded.updated_at_ms
        """,
        (
            workspace.workspace_id,
            _canonical_workspace_path(workspace.canonical_path),
            workspace.display_name,
            workspace.config_reference,
            _datetime_to_ms(workspace.created_at),
            _datetime_to_ms(workspace.updated_at),
        ),
    )


def _validate_source_checkpoint(
    connection: sqlite3.Connection,
    command: CreateJobCommand,
) -> None:
    if not command.source_checkpoint:
        return
    source_session_id = command.source_session_id
    assert source_session_id is not None
    source_exists = connection.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?",
        (source_session_id,),
    ).fetchone()
    if source_exists is None:
        raise SessionNotFoundError(source_session_id)
    owned = {
        (row[0], row[1])
        for row in connection.execute(
            "SELECT kind, value FROM provider_references WHERE session_id = ?",
            (source_session_id,),
        ).fetchall()
    }
    if any(
        (reference.kind, reference.value) not in owned for reference in command.source_checkpoint
    ):
        raise ValueError("source checkpoint contains a reference not owned by its session")


def _read_idempotency_replay(
    connection: sqlite3.Connection,
    command: CreateJobCommand,
    request_hash: str,
) -> CreateJobResult | None:
    if command.idempotency_key is None:
        return None
    row = _fetch_one_mapping(
        connection,
        """
        SELECT request_hash, job_id
        FROM idempotency_keys
        WHERE principal_id = ?
          AND workspace_id = ?
          AND command_family = ?
          AND idempotency_key = ?
          AND source_session_id IS ?
        """,
        (
            command.owner_id,
            command.workspace.workspace_id,
            command.command_family,
            command.idempotency_key,
            command.source_session_id,
        ),
    )
    if row is None:
        return None
    if row["request_hash"] != request_hash:
        raise IdempotencyConflictError(command.idempotency_key)
    job = _read_job(connection, row["job_id"])
    if job is None:
        raise StoreSchemaError("idempotency key references a missing job")
    return CreateJobResult(handle=_job_handle(job), created=False)


def _validate_session_admission(
    connection: sqlite3.Connection,
    command: CreateJobCommand,
) -> bool:
    if command.session_id is None:
        return False
    row = _fetch_one_mapping(
        connection,
        "SELECT * FROM sessions WHERE session_id = ?",
        (command.session_id,),
    )
    if row is None:
        if not command.create_session:
            raise SessionNotFoundError(command.session_id)
        if command.parent_session_id is not None:
            parent = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (command.parent_session_id,),
            ).fetchone()
            if parent is None:
                raise SessionNotFoundError(command.parent_session_id)
        return True

    expected = (
        command.workspace.workspace_id,
        command.backend_id,
        command.owner_id,
        command.access_policy,
        command.parent_session_id,
    )
    actual = (
        row["workspace_id"],
        row["backend_id"],
        row["owner_id"],
        row["access_policy"],
        row["parent_session_id"],
    )
    if actual != expected:
        raise ValueError("session identity does not match the admitted request")
    return False


def _raise_if_session_busy(
    connection: sqlite3.Connection,
    session_id: str | None,
) -> None:
    if session_id is None:
        return
    row = connection.execute(
        """
        SELECT job_id FROM jobs
        WHERE session_id = ? AND state IN ('queued', 'running', 'input_required')
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    if row is not None:
        raise SessionBusyError(session_id, row[0])


def _insert_session(
    connection: sqlite3.Connection,
    command: CreateJobCommand,
    now_ms: int,
) -> None:
    assert command.session_id is not None
    connection.execute(
        """
        INSERT INTO sessions (
          session_id, workspace_id, backend_id, owner_id, access_policy,
          parent_session_id, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            command.session_id,
            command.workspace.workspace_id,
            command.backend_id,
            command.owner_id,
            command.access_policy,
            command.parent_session_id,
            now_ms,
            now_ms,
        ),
    )


def _insert_source_checkpoint(
    connection: sqlite3.Connection,
    job_id: str,
    command: CreateJobCommand,
    now_ms: int,
) -> None:
    source_session_id = command.source_session_id
    if command.source_checkpoint:
        assert source_session_id is not None
    for reference in command.source_checkpoint:
        connection.execute(
            """
            INSERT INTO provider_references (
              provider_reference_id, backend_id, kind, value,
              session_id, job_id, attempt_number, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            (
                new_id(),
                command.backend_id,
                reference.kind,
                reference.value,
                source_session_id,
                job_id,
                now_ms,
            ),
        )


def _insert_idempotency_key(
    connection: sqlite3.Connection,
    job_id: str,
    command: CreateJobCommand,
    request_hash: str,
    now_ms: int,
) -> None:
    if command.idempotency_key is None:
        return
    connection.execute(
        """
        INSERT INTO idempotency_keys (
          idempotency_id, principal_id, workspace_id, command_family,
          idempotency_key, source_session_id, request_hash, job_id, created_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_id(),
            command.owner_id,
            command.workspace.workspace_id,
            command.command_family,
            command.idempotency_key,
            command.source_session_id,
            request_hash,
            job_id,
            now_ms,
        ),
    )


def _insert_queued_event(
    connection: sqlite3.Connection,
    job_id: str,
    backend_id: str,
    event: BackendEvent,
    now_ms: int,
) -> None:
    provider_reference_id = _ensure_job_provider_reference(
        connection,
        job_id,
        backend_id,
        event.provider_reference,
        now_ms,
    )
    connection.execute(
        """
        INSERT INTO job_events (
          job_id, sequence, event_type, payload_json, payload_schema_version,
          attempt_number, created_at_ms, provider_event_type, provider_event_id,
          provider_reference_id
        ) VALUES (?, 1, ?, ?, ?, NULL, ?, ?, NULL, ?)
        """,
        (
            job_id,
            event.type,
            _canonical_json(dict(event.payload)),
            _JSON_SCHEMA_VERSION,
            _datetime_to_ms(event.occurred_at),
            event.provider_event_type,
            provider_reference_id,
        ),
    )


def _ensure_job_provider_reference(
    connection: sqlite3.Connection,
    job_id: str,
    backend_id: str,
    reference: ProviderReference | None,
    now_ms: int,
) -> str | None:
    if reference is None:
        return None
    row = connection.execute(
        """
        SELECT provider_reference_id
        FROM provider_references
        WHERE backend_id = ? AND kind = ? AND value = ?
          AND job_id = ? AND attempt_number IS NULL
        """,
        (backend_id, reference.kind, reference.value, job_id),
    ).fetchone()
    if row is not None:
        provider_reference_id = row[0]
        if not isinstance(provider_reference_id, str) or not provider_reference_id:
            raise StoreSchemaError("job provider reference has an invalid identity")
        return provider_reference_id
    provider_reference_id = new_id()
    connection.execute(
        """
        INSERT INTO provider_references (
          provider_reference_id, backend_id, kind, value,
          session_id, job_id, attempt_number, created_at_ms
        ) VALUES (?, ?, ?, ?, NULL, ?, NULL, ?)
        """,
        (
            provider_reference_id,
            backend_id,
            reference.kind,
            reference.value,
            job_id,
            now_ms,
        ),
    )
    return provider_reference_id


def _read_session(
    connection: sqlite3.Connection,
    session_id: str,
) -> AgentSession | None:
    row = _fetch_one_mapping(
        connection,
        "SELECT * FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    if row is None:
        return None
    return AgentSession(
        session_id=row["session_id"],
        workspace_id=row["workspace_id"],
        backend_id=row["backend_id"],
        owner_id=row["owner_id"],
        access_policy=row["access_policy"],
        parent_session_id=row["parent_session_id"],
        provider_references=_read_provider_references(
            connection,
            "session_id",
            session_id,
        ),
        created_at=_ms_to_datetime(row["created_at_ms"]),
        updated_at=_ms_to_datetime(row["updated_at_ms"]),
    )


def _read_provider_references(
    connection: sqlite3.Connection,
    column: str,
    identity: str,
) -> tuple[ProviderReference, ...]:
    statements = {
        "session_id": """
            SELECT kind, value FROM provider_references
            WHERE session_id = ? ORDER BY created_at_ms, provider_reference_id
        """,
        "job_id": """
            SELECT kind, value FROM provider_references
            WHERE job_id = ? ORDER BY created_at_ms, provider_reference_id
        """,
    }
    references: list[ProviderReference] = []
    seen: set[tuple[str, str]] = set()
    for kind, value in connection.execute(statements[column], (identity,)).fetchall():
        identity_pair = (kind, value)
        if identity_pair in seen:
            continue
        seen.add(identity_pair)
        references.append(ProviderReference(kind=kind, value=value))
    return tuple(references)


def _read_job(connection: sqlite3.Connection, job_id: str) -> AgentJob | None:
    row = _fetch_one_mapping(
        connection,
        f"{_JOB_SELECT} WHERE j.job_id = ?",
        (job_id,),
    )
    return None if row is None else _job_from_row(connection, row)


def _job_from_row(connection: sqlite3.Connection, row: dict[str, Any]) -> AgentJob:
    operation = _decode_operation(
        row["operation_json"],
        row["operation_schema_version"],
        row["operation_kind"],
    )
    requested_config = _decode_model(
        row["requested_config_json"],
        row["requested_config_schema_version"],
        "requested_config_schema_version",
        RequestedExecutionConfig,
    )
    resolved_config = None
    if row["resolved_config_json"] is not None:
        resolved_config = _decode_model(
            row["resolved_config_json"],
            row["resolved_config_schema_version"],
            "resolved_config_schema_version",
            ResolvedExecutionConfig,
        )
    elif row["resolved_config_schema_version"] is not None:
        raise StoreSchemaError("resolved_config_schema_version has no typed payload")

    source_checkpoint = tuple(
        ProviderReference(kind=kind, value=value)
        for kind, value in connection.execute(
            """
            SELECT kind, value FROM provider_references
            WHERE job_id = ? AND session_id IS NOT NULL AND attempt_number IS NULL
            ORDER BY created_at_ms, provider_reference_id
            """,
            (row["job_id"],),
        ).fetchall()
    )
    lease_generation = row["lease_generation"] or None
    return AgentJob(
        job_id=row["job_id"],
        workspace_id=row["workspace_id"],
        backend_id=row["backend_id"],
        owner_id=row["owner_id"],
        operation=operation,
        requested_config=requested_config,
        request_hash=row["request_hash"],
        access_policy=row["access_policy"],
        session_id=row["session_id"],
        idempotency_key=row["idempotency_key"],
        source_checkpoint=source_checkpoint,
        state=row["state"],
        resolved_config=resolved_config,
        cancel_requested_at=_optional_ms(row["cancel_requested_at_ms"]),
        lease_owner_id=row["lease_owner"],
        lease_generation=lease_generation,
        lease_expires_at=_optional_ms(row["lease_expires_at_ms"]),
        retry_at=_optional_ms(row["retry_at_ms"]),
        created_at=_ms_to_datetime(row["created_at_ms"]),
        updated_at=_ms_to_datetime(row["updated_at_ms"]),
        completed_at=_optional_ms(row["terminal_at_ms"]),
    )


def _optional_ms(value: int | None) -> datetime | None:
    return None if value is None else _ms_to_datetime(value)


def _decode_operation(payload: str, version: int, expected_kind: str) -> AgentOperation:
    _require_json_version(version, "operation_schema_version")
    try:
        operation = _OPERATION_ADAPTER.validate_json(payload)
    except (ValidationError, ValueError) as error:
        raise StoreSchemaError("operation_json is not a valid typed operation") from error
    if operation.kind != expected_kind:
        raise StoreSchemaError("operation_kind does not match operation_json")
    return operation


def _decode_model[ModelT: BaseModel](
    payload: str,
    version: int | None,
    version_column: str,
    model: type[ModelT],
) -> ModelT:
    _require_json_version(version, version_column)
    try:
        return model.model_validate_json(payload)
    except (ValidationError, ValueError) as error:
        raise StoreSchemaError(f"{version_column} payload is invalid") from error


def _decode_json_object(payload: str, version: int, version_column: str) -> dict[str, Any]:
    _require_json_version(version, version_column)
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as error:
        raise StoreSchemaError(f"{version_column} payload is invalid JSON") from error
    if not isinstance(decoded, dict):
        raise StoreSchemaError(f"{version_column} payload is not an object")
    return decoded


def _require_json_version(version: int | None, column: str) -> None:
    if version != _JSON_SCHEMA_VERSION:
        raise StoreSchemaError(f"unsupported {column}: {version}")


def _job_handle(job: AgentJob) -> JobHandle:
    return JobHandle(
        job_id=job.job_id,
        session_id=job.session_id,
        operation=job.operation,
    )


def _list_jobs(
    connection: sqlite3.Connection,
    query: JobQuery,
    cursor: tuple[int, str] | None,
) -> StoredJobPage:
    states = tuple(sorted(query.states))
    state_placeholders = ", ".join("?" for _ in states)
    predicates = [
        "j.workspace_id = ?",
        f"j.state IN ({state_placeholders})",
        "(j.owner_id = ? OR (j.access_policy = 'workspace' AND ? = 1))",
    ]
    parameters: list[object] = [
        query.workspace_id,
        *states,
        query.access.principal_id,
        int(query.access.workspace_authorized),
    ]
    if cursor is not None:
        created_at_ms, job_id = cursor
        predicates.append("(j.created_at_ms < ? OR (j.created_at_ms = ? AND j.job_id < ?))")
        parameters.extend((created_at_ms, created_at_ms, job_id))
    parameters.append(query.limit + 1)
    rows = _fetch_all_mappings(
        connection,
        f"""
        {_JOB_SELECT}
        WHERE {" AND ".join(predicates)}
        ORDER BY j.created_at_ms DESC, j.job_id DESC
        LIMIT ?
        """,
        tuple(parameters),
    )
    page_rows = rows[: query.limit]
    jobs = tuple(_job_from_row(connection, row) for row in page_rows)
    next_cursor = None
    if len(rows) > query.limit:
        final = page_rows[-1]
        next_cursor = _encode_cursor(final["created_at_ms"], final["job_id"])
    return StoredJobPage(jobs=jobs, next_cursor=next_cursor)


def _encode_cursor(created_at_ms: int, job_id: str) -> str:
    payload = _canonical_json(
        {"v": _CURSOR_VERSION, "created_at_ms": created_at_ms, "job_id": job_id}
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> tuple[int, str]:
    if re.fullmatch(r"[A-Za-z0-9_-]+", cursor) is None:
        raise InvalidCursorError("invalid stored-job cursor")
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.b64decode(
            (cursor + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"v", "created_at_ms", "job_id"}:
            raise ValueError
        if type(payload["v"]) is not int or payload["v"] != _CURSOR_VERSION:
            raise ValueError
        if type(payload["created_at_ms"]) is not int:
            raise ValueError
        if not isinstance(payload["job_id"], str) or not payload["job_id"]:
            raise ValueError
    except (
        binascii.Error,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        UnicodeEncodeError,
        ValueError,
    ) as error:
        raise InvalidCursorError("invalid stored-job cursor") from error
    return payload["created_at_ms"], payload["job_id"]


def _read_events(
    connection: sqlite3.Connection,
    job_id: str,
    after_sequence: int,
    limit: int,
) -> EventPage:
    exists = connection.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if exists is None:
        raise JobNotFoundError(job_id)
    rows = _fetch_all_mappings(
        connection,
        """
        SELECT * FROM job_events
        WHERE job_id = ? AND sequence > ?
        ORDER BY sequence
        LIMIT ?
        """,
        (job_id, after_sequence, limit + 1),
    )
    page_rows = rows[:limit]
    events = tuple(_event_from_row(connection, row) for row in page_rows)
    return EventPage(
        events=events,
        next_after_sequence=None if not events else events[-1].sequence,
        has_more=len(rows) > limit,
    )


def _event_from_row(connection: sqlite3.Connection, row: dict[str, Any]) -> JobEvent:
    payload = _decode_json_object(
        row["payload_json"],
        row["payload_schema_version"],
        "payload_schema_version",
    )
    provider_reference = _read_event_provider_reference(connection, row)
    try:
        return JobEvent(
            job_id=row["job_id"],
            sequence=row["sequence"],
            type=row["event_type"],
            payload=payload,
            payload_schema_version=row["payload_schema_version"],
            occurred_at=_ms_to_datetime(row["created_at_ms"]),
            attempt_number=row["attempt_number"],
            provider_event_type=row["provider_event_type"],
            provider_reference=provider_reference,
        )
    except ValidationError as error:
        raise StoreSchemaError("job event row is invalid") from error


def _read_event_provider_reference(
    connection: sqlite3.Connection,
    event_row: dict[str, Any],
) -> ProviderReference | None:
    provider_reference_id = event_row["provider_reference_id"]
    if provider_reference_id is None:
        return None
    row = connection.execute(
        """
        SELECT kind, value, job_id
        FROM provider_references
        WHERE provider_reference_id = ?
        """,
        (provider_reference_id,),
    ).fetchone()
    if row is None or row[2] != event_row["job_id"]:
        raise StoreSchemaError("job event provider reference is missing or belongs to another job")
    try:
        return ProviderReference(kind=row[0], value=row[1])
    except ValidationError as error:
        raise StoreSchemaError("job event provider reference row is invalid") from error


def _apply_migrations(connection: sqlite3.Connection) -> None:
    """Validate and apply all known migrations in one serialized transaction."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        applied = _read_applied_migrations(connection)
        _validate_applied_migrations(applied)
        for migration in MIGRATIONS:
            if migration.migration_id in applied:
                continue
            _apply_migration(connection, migration)
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _read_applied_migrations(connection: sqlite3.Connection) -> dict[str, str]:
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
    ).fetchone()
    if table_exists is None:
        return {}
    rows = connection.execute(
        "SELECT migration_id, checksum FROM schema_migrations ORDER BY rowid"
    ).fetchall()
    return dict(rows)


def _validate_applied_migrations(applied: dict[str, str]) -> None:
    known = {migration.migration_id: migration for migration in MIGRATIONS}
    unknown = set(applied).difference(known)
    if unknown:
        migration_ids = ", ".join(sorted(unknown))
        raise RuntimeError(f"database schema is newer than this Nexus MCP build: {migration_ids}")

    expected_prefix = tuple(migration.migration_id for migration in MIGRATIONS[: len(applied)])
    if tuple(applied) != expected_prefix:
        raise RuntimeError("database migration history is not a valid forward-only prefix")

    for migration_id, checksum in applied.items():
        if checksum != known[migration_id].checksum:
            raise RuntimeError(f"migration checksum mismatch for {migration_id}")


def _apply_migration(connection: sqlite3.Connection, migration: Migration) -> None:
    for statement in migration.statements:
        connection.execute(statement)
    connection.execute(
        "INSERT INTO schema_migrations (migration_id, checksum, applied_at_ms) VALUES (?, ?, ?)",
        (migration.migration_id, migration.checksum, time.time_ns() // 1_000_000),
    )
