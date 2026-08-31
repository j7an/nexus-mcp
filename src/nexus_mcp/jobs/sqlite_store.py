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
from typing import Any, TypeVar, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from nexus_mcp.core import (
    TERMINAL_STATES,
    AgentJob,
    AgentOperation,
    AgentSession,
    BackendEvent,
    CancelReceipt,
    IdempotencyConflictError,
    InputAlreadyResolvedError,
    InputNotFoundError,
    InputRequest,
    InputResolutionReceipt,
    InputResponse,
    JobAttempt,
    JobError,
    JobEvent,
    JobHandle,
    JobNotFoundError,
    JobPhase,
    JobResultEnvelope,
    JobState,
    OperationResult,
    PendingInput,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    SessionBusyError,
    SessionNotFoundError,
    StaleLeaseError,
    Workspace,
    WorkspaceInvalidError,
    WorkspaceSelector,
    new_id,
    validate_job_transition,
)
from nexus_mcp.jobs.migrations import MIGRATIONS, Migration
from nexus_mcp.jobs.paths import default_database_path
from nexus_mcp.jobs.store import (
    CancelJobCommand,
    CancelledTerminalOutcome,
    ClaimedJob,
    ControlSnapshot,
    CreateJobCommand,
    CreateJobResult,
    EventPage,
    FailedTerminalOutcome,
    JobQuery,
    LeaseToken,
    PrunePolicy,
    PruneResult,
    ResolveInputCommand,
    RuntimeLease,
    RuntimeLeaseBusyError,
    StoredJobPage,
    SucceededTerminalOutcome,
    TerminalOutcome,
)

__all__ = ["InvalidCursorError", "SQLiteJobStore", "StoreSchemaError"]

_ResultT = TypeVar("_ResultT")
_JSON_SCHEMA_VERSION = 1
_CURSOR_VERSION = 1
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_OPERATION_ADAPTER: TypeAdapter[AgentOperation] = TypeAdapter(AgentOperation)
_INPUT_REQUEST_ADAPTER: TypeAdapter[InputRequest] = TypeAdapter(InputRequest)
_INPUT_RESPONSE_ADAPTER: TypeAdapter[InputResponse] = TypeAdapter(InputResponse)
_OPERATION_RESULT_ADAPTER: TypeAdapter[OperationResult] = TypeAdapter(OperationResult)
_EVENT_WATERMARK_TYPE = "__pruned_event_watermark__"
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

    async def get_job_result(self, job_id: str) -> JobResultEnvelope | JobError | None:
        """Return one normalized durable terminal outcome."""
        return await self._worker._call(lambda connection: _read_job_result(connection, job_id))

    async def get_job_attempts(self, job_id: str) -> tuple[JobAttempt, ...]:
        """Return persisted attempts in ascending order."""
        return await self._worker._call(lambda connection: _read_job_attempts(connection, job_id))

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

    async def get_pending_inputs(self, job_id: str) -> tuple[PendingInput, ...]:
        """Return unresolved durable input requests."""
        return await self._worker._call(
            lambda connection: _read_pending_inputs(connection, job_id, unresolved_only=True)
        )

    async def list_jobs(self, query: JobQuery) -> StoredJobPage:
        """Return one authorized descending keyset page of typed job snapshots."""
        cursor = None if query.cursor is None else _decode_cursor(query.cursor)
        return await self._worker._call(lambda connection: _list_jobs(connection, query, cursor))

    async def claim_next(
        self,
        owner_id: str,
        lease_until: datetime,
        *,
        event: BackendEvent,
    ) -> ClaimedJob | None:
        """Claim one eligible job under a new fencing generation."""
        lease_until = _normalize_datetime(lease_until)
        return await self._worker._call(
            lambda connection: _claim_next_transaction(
                connection,
                owner_id,
                lease_until,
                event,
            )
        )

    async def renew_lease(self, token: LeaseToken, lease_until: datetime) -> bool:
        """Renew only the current job fence."""
        lease_until = _normalize_datetime(lease_until)
        return await self._worker._call(
            lambda connection: _renew_lease_transaction(connection, token, lease_until)
        )

    async def store_resolved_config(
        self, token: LeaseToken, config: ResolvedExecutionConfig
    ) -> None:
        """Persist effective configuration under one worker fence."""
        config_json = _canonical_json(config)
        await self._worker._call(
            lambda connection: _store_resolved_config_transaction(
                connection,
                token,
                config,
                config_json,
            )
        )

    async def record_provider_reference(
        self, token: LeaseToken, reference: ProviderReference
    ) -> None:
        """Persist one provider reference under one worker fence."""
        await self._worker._call(
            lambda connection: _record_provider_reference_transaction(
                connection,
                token,
                reference,
            )
        )

    async def append_events(
        self, token: LeaseToken, events: tuple[BackendEvent, ...]
    ) -> tuple[JobEvent, ...]:
        """Append events under one worker fence."""
        return await self._worker._call(
            lambda connection: _append_events_transaction(connection, token, events)
        )

    async def mark_input_required(
        self,
        token: LeaseToken,
        inputs: tuple[PendingInput, ...],
        *,
        event: BackendEvent,
    ) -> None:
        """Persist pending inputs and their semantic transition event."""
        if not inputs:
            raise ValueError("at least one pending input is required")
        input_ids = tuple(item.input_id for item in inputs)
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("pending input ids must be unique")
        if any(item.job_id != token.job_id for item in inputs):
            raise ValueError("pending input belongs to another job")
        await self._worker._call(
            lambda connection: _mark_input_required_transaction(
                connection,
                token,
                inputs,
                event,
            )
        )

    async def mark_running(
        self,
        token: LeaseToken,
        resolved_input_ids: tuple[str, ...],
        *,
        event: BackendEvent,
    ) -> None:
        """Enter or resume running under one worker fence."""
        if len(resolved_input_ids) != len(set(resolved_input_ids)):
            raise ValueError("resolved input ids must be unique")
        await self._worker._call(
            lambda connection: _mark_running_transaction(
                connection,
                token,
                resolved_input_ids,
                event,
            )
        )

    async def mark_reconciling(
        self,
        token: LeaseToken,
        error: JobError,
        *,
        event: BackendEvent,
    ) -> None:
        """Persist an uncertain-provider reconciliation checkpoint."""
        error_json = _canonical_json(error)
        await self._worker._call(
            lambda connection: _mark_reconciling_transaction(
                connection,
                token,
                error_json,
                error.code,
                event,
            )
        )

    async def schedule_retry(
        self,
        token: LeaseToken,
        retry_at: datetime,
        error: JobError,
        *,
        event: BackendEvent,
    ) -> None:
        """Close one attempt and schedule a safe retry."""
        retry_at = _normalize_datetime(retry_at)
        error_json = _canonical_json(error)
        await self._worker._call(
            lambda connection: _schedule_retry_transaction(
                connection,
                token,
                retry_at,
                error_json,
                error.retry_disposition,
                event,
            )
        )

    async def get_control_snapshot(self, token: LeaseToken) -> ControlSnapshot:
        """Read worker controls under one fence."""
        return await self._worker._call(
            lambda connection: _read_control_snapshot_transaction(connection, token)
        )

    async def resolve_input(self, command: ResolveInputCommand) -> InputResolutionReceipt:
        """Resolve one pending input idempotently."""
        return await self._worker._call(
            lambda connection: _resolve_input_transaction(connection, command)
        )

    async def request_cancel(self, command: CancelJobCommand) -> CancelReceipt:
        """Persist cancellation intent idempotently."""
        return await self._worker._call(
            lambda connection: _request_cancel_transaction(connection, command)
        )

    async def terminalize(
        self,
        token: LeaseToken,
        outcome: TerminalOutcome,
        *,
        event: BackendEvent,
    ) -> AgentJob:
        """Commit one fenced terminal outcome and semantic event."""
        return await self._worker._call(
            lambda connection: _terminalize_transaction(
                connection,
                token,
                outcome,
                event,
            )
        )

    async def read_events(self, job_id: str, after_sequence: int, limit: int) -> EventPage:
        """Read committed admission events after one bounded job-local sequence."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if not 1 <= limit <= 1000:
            raise ValueError("event page limit must be from 1 through 1000")
        return await self._worker._call(
            lambda connection: _read_events(connection, job_id, after_sequence, limit)
        )

    async def acquire_runtime_lease(
        self, runtime_key: str, owner_id: str, lease_until: datetime
    ) -> RuntimeLease:
        """Acquire or generation-fence one managed runtime."""
        lease_until = _normalize_datetime(lease_until)
        return await self._worker._call(
            lambda connection: _acquire_runtime_lease_transaction(
                connection,
                runtime_key,
                owner_id,
                lease_until,
            )
        )

    async def renew_runtime_lease(self, lease: RuntimeLease, lease_until: datetime) -> bool:
        """Renew the matching live runtime generation."""
        lease_until = _normalize_datetime(lease_until)
        return await self._worker._call(
            lambda connection: _renew_runtime_lease_transaction(
                connection,
                lease,
                lease_until,
            )
        )

    async def release_runtime_lease(self, lease: RuntimeLease) -> None:
        """Release only the matching live runtime generation."""
        await self._worker._call(
            lambda connection: _release_runtime_lease_transaction(connection, lease)
        )

    async def prune(self, policy: PrunePolicy, now: datetime) -> PruneResult:
        """Prune eligible terminal history atomically."""
        now = _normalize_datetime(now)
        return await self._worker._call(
            lambda connection: _prune_transaction(connection, policy, now)
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


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be a timezone-aware UTC datetime")
    return value.astimezone(UTC)


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _run_immediate[TransactionT](
    connection: sqlite3.Connection,
    operation: Callable[[], TransactionT],
) -> TransactionT:
    connection.execute("BEGIN IMMEDIATE")
    try:
        result = operation()
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _require_worker_job(
    connection: sqlite3.Connection,
    token: LeaseToken,
    now_ms: int,
) -> dict[str, Any]:
    row = _fetch_one_mapping(
        connection,
        """
        SELECT j.*
        FROM jobs AS j
        WHERE j.job_id = ?
          AND j.lease_owner = ?
          AND j.lease_generation = ?
          AND j.lease_expires_at_ms > ?
          AND j.state NOT IN ('completed', 'failed', 'cancelled')
          AND EXISTS (
            SELECT 1 FROM job_attempts AS a
            WHERE a.job_id = j.job_id
              AND a.attempt_number = ?
              AND a.owner_id = ?
              AND a.lease_generation = ?
          )
        """,
        (
            token.job_id,
            token.owner_id,
            token.generation,
            now_ms,
            token.attempt_number,
            token.owner_id,
            token.generation,
        ),
    )
    if row is None:
        raise StaleLeaseError(token.job_id, token.generation)
    return row


def _touch_worker_job(
    connection: sqlite3.Connection,
    token: LeaseToken,
    now_ms: int,
) -> None:
    cursor = connection.execute(
        """
        UPDATE jobs
        SET updated_at_ms = ?
        WHERE job_id = ?
          AND lease_owner = ?
          AND lease_generation = ?
          AND lease_expires_at_ms > ?
          AND state NOT IN ('completed', 'failed', 'cancelled')
          AND EXISTS (
            SELECT 1 FROM job_attempts AS a
            WHERE a.job_id = jobs.job_id
              AND a.attempt_number = ?
              AND a.owner_id = ?
              AND a.lease_generation = ?
          )
        """,
        (
            now_ms,
            token.job_id,
            token.owner_id,
            token.generation,
            now_ms,
            token.attempt_number,
            token.owner_id,
            token.generation,
        ),
    )
    if cursor.rowcount != 1:
        raise StaleLeaseError(token.job_id, token.generation)


def _update_worker_attempt(
    connection: sqlite3.Connection,
    token: LeaseToken,
    now_ms: int,
    assignments: str,
    values: tuple[object, ...],
) -> None:
    cursor = connection.execute(
        f"""
        UPDATE job_attempts
        SET {assignments}
        WHERE job_id = ?
          AND attempt_number = ?
          AND owner_id = ?
          AND lease_generation = ?
          AND EXISTS (
            SELECT 1 FROM jobs AS j
            WHERE j.job_id = job_attempts.job_id
              AND j.lease_owner = ?
              AND j.lease_generation = ?
              AND j.lease_expires_at_ms > ?
              AND j.state NOT IN ('completed', 'failed', 'cancelled')
          )
        """,
        (
            *values,
            token.job_id,
            token.attempt_number,
            token.owner_id,
            token.generation,
            token.owner_id,
            token.generation,
            now_ms,
        ),
    )
    if cursor.rowcount != 1:
        raise StaleLeaseError(token.job_id, token.generation)


def _claim_next_transaction(
    connection: sqlite3.Connection,
    owner_id: str,
    lease_until: datetime,
    event: BackendEvent,
) -> ClaimedJob | None:
    def claim() -> ClaimedJob | None:
        now_ms = _now_ms()
        row = _fetch_one_mapping(
            connection,
            """
            SELECT * FROM jobs
            WHERE state NOT IN ('completed', 'failed', 'cancelled')
              AND (lease_expires_at_ms IS NULL OR lease_expires_at_ms <= ?)
              AND (retry_at_ms IS NULL OR retry_at_ms <= ?)
            ORDER BY created_at_ms, job_id
            LIMIT 1
            """,
            (now_ms, now_ms),
        )
        if row is None:
            return None

        job_id = row["job_id"]
        generation = int(row["lease_generation"]) + 1
        attempt_number = (
            int(
                connection.execute(
                    "SELECT count(*) FROM job_attempts WHERE job_id = ?",
                    (job_id,),
                ).fetchone()[0]
            )
            + 1
        )
        connection.execute(
            """
            UPDATE job_attempts
            SET ended_at_ms = ?
            WHERE job_id = ? AND ended_at_ms IS NULL
            """,
            (now_ms, job_id),
        )
        cursor = connection.execute(
            """
            UPDATE jobs
            SET lease_owner = ?, lease_generation = ?, lease_expires_at_ms = ?,
                retry_at_ms = NULL, updated_at_ms = ?
            WHERE job_id = ?
              AND state NOT IN ('completed', 'failed', 'cancelled')
              AND (lease_expires_at_ms IS NULL OR lease_expires_at_ms <= ?)
              AND (retry_at_ms IS NULL OR retry_at_ms <= ?)
            """,
            (
                owner_id,
                generation,
                _datetime_to_ms(lease_until),
                now_ms,
                job_id,
                now_ms,
                now_ms,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleLeaseError(job_id, generation)
        phase: JobPhase = "reconciling" if row["lease_generation"] else "claiming"
        connection.execute(
            """
            INSERT INTO job_attempts (
              job_id, attempt_number, phase, owner_id, lease_generation,
              lease_expires_at_ms, heartbeat_at_ms, started_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                attempt_number,
                phase,
                owner_id,
                generation,
                _datetime_to_ms(lease_until),
                now_ms,
                now_ms,
            ),
        )
        _insert_job_event(connection, job_id, event, attempt_number, now_ms)
        job = _read_job(connection, job_id)
        if job is None:
            raise StoreSchemaError("claimed job disappeared inside its transaction")
        attempt = JobAttempt(
            job_id=job_id,
            attempt_number=attempt_number,
            phase=phase,
            worker_id=owner_id,
            lease_generation=generation,
            lease_expires_at=lease_until,
            heartbeat_at=_ms_to_datetime(now_ms),
            started_at=_ms_to_datetime(now_ms),
        )
        return ClaimedJob(
            job=job,
            attempt=attempt,
            token=LeaseToken(
                job_id=job_id,
                owner_id=owner_id,
                generation=generation,
                attempt_number=attempt_number,
            ),
        )

    return _run_immediate(connection, claim)


def _renew_lease_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    lease_until: datetime,
) -> bool:
    def renew() -> bool:
        now_ms = _now_ms()
        cursor = connection.execute(
            """
            UPDATE jobs
            SET lease_expires_at_ms = ?, updated_at_ms = ?
            WHERE job_id = ?
              AND lease_owner = ?
              AND lease_generation = ?
              AND lease_expires_at_ms > ?
              AND state NOT IN ('completed', 'failed', 'cancelled')
              AND EXISTS (
                SELECT 1 FROM job_attempts AS a
                WHERE a.job_id = jobs.job_id
                  AND a.attempt_number = ?
                  AND a.owner_id = ?
                  AND a.lease_generation = ?
              )
            """,
            (
                _datetime_to_ms(lease_until),
                now_ms,
                token.job_id,
                token.owner_id,
                token.generation,
                now_ms,
                token.attempt_number,
                token.owner_id,
                token.generation,
            ),
        )
        if cursor.rowcount != 1:
            return False
        attempt_cursor = connection.execute(
            """
            UPDATE job_attempts
            SET lease_expires_at_ms = ?, heartbeat_at_ms = ?
            WHERE job_id = ? AND attempt_number = ?
              AND owner_id = ? AND lease_generation = ?
              AND ended_at_ms IS NULL
            """,
            (
                _datetime_to_ms(lease_until),
                now_ms,
                token.job_id,
                token.attempt_number,
                token.owner_id,
                token.generation,
            ),
        )
        if attempt_cursor.rowcount != 1:
            raise StoreSchemaError("current job lease has no matching active attempt")
        return True

    return _run_immediate(connection, renew)


def _store_resolved_config_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    config: ResolvedExecutionConfig,
    config_json: str,
) -> None:
    def store() -> None:
        now_ms = _now_ms()
        row = _require_worker_job(connection, token, now_ms)
        if row["resolved_config_json"] is not None:
            existing = _decode_model(
                row["resolved_config_json"],
                row["resolved_config_schema_version"],
                "resolved_config_schema_version",
                ResolvedExecutionConfig,
            )
            if existing != config:
                raise ValueError("resolved execution config is already stored")
        cursor = connection.execute(
            """
            UPDATE jobs
            SET resolved_config_json = ?, resolved_config_schema_version = ?, updated_at_ms = ?
            WHERE job_id = ? AND lease_owner = ? AND lease_generation = ?
              AND lease_expires_at_ms > ?
              AND state NOT IN ('completed', 'failed', 'cancelled')
            """,
            (
                config_json,
                _JSON_SCHEMA_VERSION,
                now_ms,
                token.job_id,
                token.owner_id,
                token.generation,
                now_ms,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleLeaseError(token.job_id, token.generation)

    _run_immediate(connection, store)


def _record_provider_reference_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    reference: ProviderReference,
) -> None:
    def record() -> None:
        now_ms = _now_ms()
        row = _require_worker_job(connection, token, now_ms)
        _touch_worker_job(connection, token, now_ms)
        connection.execute(
            """
            INSERT OR IGNORE INTO provider_references (
              provider_reference_id, backend_id, kind, value,
              session_id, job_id, attempt_number, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                row["backend_id"],
                reference.kind,
                reference.value,
                row["session_id"],
                token.job_id,
                token.attempt_number,
                now_ms,
            ),
        )
        if row["session_id"] is not None:
            cursor = connection.execute(
                """
                UPDATE sessions
                SET updated_at_ms = ?
                WHERE session_id = ?
                  AND EXISTS (
                    SELECT 1 FROM jobs
                    WHERE job_id = ? AND lease_owner = ? AND lease_generation = ?
                      AND lease_expires_at_ms > ?
                  )
                """,
                (
                    now_ms,
                    row["session_id"],
                    token.job_id,
                    token.owner_id,
                    token.generation,
                    now_ms,
                ),
            )
            if cursor.rowcount != 1:
                raise StaleLeaseError(token.job_id, token.generation)

    _run_immediate(connection, record)


def _append_events_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    events: tuple[BackendEvent, ...],
) -> tuple[JobEvent, ...]:
    def append() -> tuple[JobEvent, ...]:
        now_ms = _now_ms()
        _require_worker_job(connection, token, now_ms)
        if events:
            _touch_worker_job(connection, token, now_ms)
        return tuple(
            _insert_job_event(connection, token.job_id, event, token.attempt_number, now_ms)
            for event in events
        )

    return _run_immediate(connection, append)


def _mark_input_required_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    inputs: tuple[PendingInput, ...],
    event: BackendEvent,
) -> None:
    def mark() -> None:
        now_ms = _now_ms()
        job = _require_worker_job(connection, token, now_ms)
        validate_job_transition(job["state"], "input_required")
        _touch_worker_job(connection, token, now_ms)
        for item in inputs:
            if (
                connection.execute(
                    "SELECT 1 FROM pending_inputs WHERE input_id = ?",
                    (item.input_id,),
                ).fetchone()
                is not None
            ):
                raise ValueError(f"pending input already exists: {item.input_id}")
            provider_reference_id = _ensure_job_provider_reference(
                connection,
                token.job_id,
                job["backend_id"],
                item.provider_reference,
                now_ms,
            )
            connection.execute(
                """
                INSERT INTO pending_inputs (
                  input_id, job_id, kind, request_json, request_schema_version,
                  response_json, response_schema_version, status,
                  provider_reference_id, created_at_ms, resolved_at_ms
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, 'pending', ?, ?, NULL)
                """,
                (
                    item.input_id,
                    item.job_id,
                    item.request.kind,
                    _canonical_json(item.request),
                    _JSON_SCHEMA_VERSION,
                    provider_reference_id,
                    _datetime_to_ms(item.created_at),
                ),
            )
        _update_worker_attempt(
            connection,
            token,
            now_ms,
            "phase = 'executing'",
            (),
        )
        _insert_job_event(connection, token.job_id, event, token.attempt_number, now_ms)
        cursor = connection.execute(
            """
            UPDATE jobs
            SET state = 'input_required', updated_at_ms = ?
            WHERE job_id = ? AND lease_owner = ? AND lease_generation = ?
              AND lease_expires_at_ms > ?
              AND state = ?
            """,
            (
                now_ms,
                token.job_id,
                token.owner_id,
                token.generation,
                now_ms,
                job["state"],
            ),
        )
        if cursor.rowcount != 1:
            raise StaleLeaseError(token.job_id, token.generation)

    _run_immediate(connection, mark)


def _mark_running_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    resolved_input_ids: tuple[str, ...],
    event: BackendEvent,
) -> None:
    def mark() -> None:
        now_ms = _now_ms()
        job = _require_worker_job(connection, token, now_ms)
        validate_job_transition(job["state"], "running")
        for input_id in resolved_input_ids:
            row = connection.execute(
                """
                SELECT status, response_json
                FROM pending_inputs
                WHERE job_id = ? AND input_id = ?
                """,
                (token.job_id, input_id),
            ).fetchone()
            if row is None or row[0] != "resolved" or row[1] is None:
                raise InputNotFoundError(token.job_id, input_id)
        if job["state"] == "input_required":
            unresolved = connection.execute(
                """
                SELECT 1 FROM pending_inputs
                WHERE job_id = ? AND status = 'pending'
                LIMIT 1
                """,
                (token.job_id,),
            ).fetchone()
            if unresolved is not None:
                raise ValueError("all pending inputs must be resolved before resuming")
        _touch_worker_job(connection, token, now_ms)
        _update_worker_attempt(
            connection,
            token,
            now_ms,
            "phase = 'executing'",
            (),
        )
        _insert_job_event(connection, token.job_id, event, token.attempt_number, now_ms)
        cursor = connection.execute(
            """
            UPDATE jobs
            SET state = 'running', retry_at_ms = NULL, updated_at_ms = ?
            WHERE job_id = ? AND lease_owner = ? AND lease_generation = ?
              AND lease_expires_at_ms > ?
              AND state = ?
            """,
            (
                now_ms,
                token.job_id,
                token.owner_id,
                token.generation,
                now_ms,
                job["state"],
            ),
        )
        if cursor.rowcount != 1:
            raise StaleLeaseError(token.job_id, token.generation)

    _run_immediate(connection, mark)


def _mark_reconciling_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    error_json: str,
    reconciliation_classification: str,
    event: BackendEvent,
) -> None:
    def mark() -> None:
        now_ms = _now_ms()
        _require_worker_job(connection, token, now_ms)
        _touch_worker_job(connection, token, now_ms)
        _update_worker_attempt(
            connection,
            token,
            now_ms,
            """
            phase = 'reconciling', error_json = ?, error_schema_version = ?,
            reconciliation_classification = ?
            """,
            (error_json, _JSON_SCHEMA_VERSION, reconciliation_classification),
        )
        _insert_job_event(connection, token.job_id, event, token.attempt_number, now_ms)

    _run_immediate(connection, mark)


def _schedule_retry_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    retry_at: datetime,
    error_json: str,
    retry_classification: str,
    event: BackendEvent,
) -> None:
    def schedule() -> None:
        now_ms = _now_ms()
        _require_worker_job(connection, token, now_ms)
        _touch_worker_job(connection, token, now_ms)
        _update_worker_attempt(
            connection,
            token,
            now_ms,
            """
            phase = 'finalizing', ended_at_ms = ?, error_json = ?, error_schema_version = ?,
            retry_classification = ?
            """,
            (now_ms, error_json, _JSON_SCHEMA_VERSION, retry_classification),
        )
        _insert_job_event(connection, token.job_id, event, token.attempt_number, now_ms)
        cursor = connection.execute(
            """
            UPDATE jobs
            SET lease_owner = NULL, lease_expires_at_ms = NULL,
                retry_at_ms = ?, updated_at_ms = ?
            WHERE job_id = ? AND lease_owner = ? AND lease_generation = ?
              AND lease_expires_at_ms > ?
              AND state NOT IN ('completed', 'failed', 'cancelled')
            """,
            (
                _datetime_to_ms(retry_at),
                now_ms,
                token.job_id,
                token.owner_id,
                token.generation,
                now_ms,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleLeaseError(token.job_id, token.generation)

    _run_immediate(connection, schedule)


def _read_control_snapshot_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
) -> ControlSnapshot:
    connection.execute("BEGIN")
    try:
        row = _require_worker_job(connection, token, _now_ms())
        unresolved = _read_pending_inputs(connection, token.job_id, unresolved_only=True)
        snapshot = ControlSnapshot(
            state=row["state"],
            cancel_requested=row["cancel_requested_at_ms"] is not None,
            unresolved_inputs=unresolved,
            lease_generation=token.generation,
        )
        connection.execute("COMMIT")
        return snapshot
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise


def _resolve_input_transaction(
    connection: sqlite3.Connection,
    command: ResolveInputCommand,
) -> InputResolutionReceipt:
    def resolve() -> InputResolutionReceipt:
        job = _fetch_one_mapping(
            connection,
            "SELECT * FROM jobs WHERE job_id = ?",
            (command.job_id,),
        )
        if job is None:
            raise JobNotFoundError(command.job_id)
        if job["state"] != "input_required":
            raise InputNotFoundError(command.job_id, command.input_id)
        row = _fetch_one_mapping(
            connection,
            "SELECT * FROM pending_inputs WHERE job_id = ? AND input_id = ?",
            (command.job_id, command.input_id),
        )
        if row is None:
            raise InputNotFoundError(command.job_id, command.input_id)
        pending = _pending_input_from_row(connection, row)
        response = pending.validate_response(command.response)
        if pending.response is not None:
            if pending.response != response:
                raise InputAlreadyResolvedError(command.job_id, command.input_id)
            return InputResolutionReceipt(
                job_id=command.job_id,
                input_id=command.input_id,
                replayed=True,
            )
        cursor = connection.execute(
            """
            UPDATE pending_inputs
            SET response_json = ?, response_schema_version = ?,
                status = 'resolved', resolved_at_ms = ?
            WHERE job_id = ? AND input_id = ? AND status = 'pending'
              AND EXISTS (
                SELECT 1 FROM jobs
                WHERE job_id = ? AND state = 'input_required'
              )
            """,
            (
                _canonical_json(response),
                _JSON_SCHEMA_VERSION,
                _datetime_to_ms(command.resolved_at),
                command.job_id,
                command.input_id,
                command.job_id,
            ),
        )
        if cursor.rowcount != 1:
            raise InputNotFoundError(command.job_id, command.input_id)
        attempt_number = _current_attempt_number(connection, command.job_id)
        _insert_job_event(
            connection,
            command.job_id,
            command.event,
            attempt_number,
            _now_ms(),
        )
        return InputResolutionReceipt(job_id=command.job_id, input_id=command.input_id)

    return _run_immediate(connection, resolve)


def _current_attempt_number(connection: sqlite3.Connection, job_id: str) -> int | None:
    row = connection.execute(
        "SELECT max(attempt_number) FROM job_attempts WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    return None if row is None or row[0] is None else int(row[0])


def _request_cancel_transaction(
    connection: sqlite3.Connection,
    command: CancelJobCommand,
) -> CancelReceipt:
    def cancel() -> CancelReceipt:
        row = _fetch_one_mapping(
            connection,
            "SELECT * FROM jobs WHERE job_id = ?",
            (command.job_id,),
        )
        if row is None:
            raise JobNotFoundError(command.job_id)
        if row["state"] in TERMINAL_STATES:
            return CancelReceipt(
                job_id=command.job_id,
                state=row["state"],
                cancel_requested=row["cancel_requested_at_ms"] is not None,
                completed_immediately=row["state"] == "cancelled",
            )
        if row["cancel_requested_at_ms"] is not None:
            return CancelReceipt(
                job_id=command.job_id,
                state=row["state"],
                cancel_requested=True,
                completed_immediately=False,
            )

        requested_at_ms = _datetime_to_ms(command.requested_at)
        completed_immediately = row["state"] == "queued"
        state: JobState
        if completed_immediately:
            validate_job_transition(row["state"], "cancelled")
            connection.execute(
                """
                UPDATE job_attempts
                SET phase = 'finalizing', ended_at_ms = ?
                WHERE job_id = ? AND ended_at_ms IS NULL
                """,
                (requested_at_ms, command.job_id),
            )
        _insert_job_event(
            connection,
            command.job_id,
            command.event,
            _current_attempt_number(connection, command.job_id),
            _now_ms(),
        )
        if completed_immediately:
            connection.execute(
                """
                INSERT INTO job_results (
                  job_id, outcome_kind, payload_json, payload_schema_version,
                  error_json, error_schema_version, created_at_ms
                ) VALUES (?, 'cancelled', NULL, NULL, NULL, NULL, ?)
                """,
                (command.job_id, requested_at_ms),
            )
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state = 'cancelled', cancel_requested_at_ms = ?,
                    lease_owner = NULL, lease_expires_at_ms = NULL,
                    retry_at_ms = NULL, updated_at_ms = ?, terminal_at_ms = ?
                WHERE job_id = ? AND state = 'queued' AND cancel_requested_at_ms IS NULL
                """,
                (
                    requested_at_ms,
                    requested_at_ms,
                    requested_at_ms,
                    command.job_id,
                ),
            )
            state = "cancelled"
        else:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET cancel_requested_at_ms = ?, updated_at_ms = ?
                WHERE job_id = ?
                  AND state IN ('running', 'input_required')
                  AND cancel_requested_at_ms IS NULL
                """,
                (requested_at_ms, requested_at_ms, command.job_id),
            )
            state = cast("JobState", row["state"])
        if cursor.rowcount != 1:
            raise JobNotFoundError(command.job_id)
        return CancelReceipt(
            job_id=command.job_id,
            state=state,
            cancel_requested=True,
            completed_immediately=completed_immediately,
        )

    return _run_immediate(connection, cancel)


def _terminalize_transaction(
    connection: sqlite3.Connection,
    token: LeaseToken,
    outcome: TerminalOutcome,
    event: BackendEvent,
) -> AgentJob:
    def terminalize() -> AgentJob:
        now_ms = _now_ms()
        row = _require_worker_job(connection, token, now_ms)
        target: JobState
        payload_json: str | None
        payload_version: int | None
        error_json: str | None
        error_version: int | None
        match outcome:
            case SucceededTerminalOutcome(result=result, completed_at=completed_at):
                operation = _decode_operation(
                    row["operation_json"],
                    row["operation_schema_version"],
                    row["operation_kind"],
                )
                if result.kind != operation.kind:
                    raise ValueError("terminal result kind does not match job operation")
                target = "completed"
                payload_json = _canonical_json(result)
                payload_version = _JSON_SCHEMA_VERSION
                error_json = None
                error_version = None
            case FailedTerminalOutcome(error=error, completed_at=completed_at):
                target = "failed"
                payload_json = None
                payload_version = None
                error_json = _canonical_json(error)
                error_version = _JSON_SCHEMA_VERSION
            case CancelledTerminalOutcome(completed_at=completed_at):
                target = "cancelled"
                payload_json = None
                payload_version = None
                error_json = None
                error_version = None
        validate_job_transition(cast("JobState", row["state"]), target)
        completed_at_ms = _datetime_to_ms(completed_at)
        _touch_worker_job(connection, token, now_ms)
        attempt_assignments = "phase = 'finalizing', ended_at_ms = ?"
        attempt_values: tuple[object, ...] = (completed_at_ms,)
        if error_json is not None:
            attempt_assignments += ", error_json = ?, error_schema_version = ?"
            attempt_values += (error_json, _JSON_SCHEMA_VERSION)
        _update_worker_attempt(
            connection,
            token,
            now_ms,
            attempt_assignments,
            attempt_values,
        )
        connection.execute(
            """
            INSERT INTO job_results (
              job_id, outcome_kind, payload_json, payload_schema_version,
              error_json, error_schema_version, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token.job_id,
                outcome.kind,
                payload_json,
                payload_version,
                error_json,
                error_version,
                completed_at_ms,
            ),
        )
        _insert_job_event(connection, token.job_id, event, token.attempt_number, now_ms)
        cursor = connection.execute(
            """
            UPDATE jobs
            SET state = ?, lease_owner = NULL, lease_expires_at_ms = NULL,
                retry_at_ms = NULL, updated_at_ms = ?, terminal_at_ms = ?
            WHERE job_id = ? AND lease_owner = ? AND lease_generation = ?
              AND lease_expires_at_ms > ?
              AND state = ?
            """,
            (
                target,
                completed_at_ms,
                completed_at_ms,
                token.job_id,
                token.owner_id,
                token.generation,
                now_ms,
                row["state"],
            ),
        )
        if cursor.rowcount != 1:
            raise StaleLeaseError(token.job_id, token.generation)
        job = _read_job(connection, token.job_id)
        if job is None:
            raise StoreSchemaError("terminalized job disappeared inside its transaction")
        return job

    return _run_immediate(connection, terminalize)


def _acquire_runtime_lease_transaction(
    connection: sqlite3.Connection,
    runtime_key: str,
    owner_id: str,
    lease_until: datetime,
) -> RuntimeLease:
    def acquire() -> RuntimeLease:
        now_ms = _now_ms()
        row = _fetch_one_mapping(
            connection,
            "SELECT * FROM runtime_leases WHERE runtime_key = ?",
            (runtime_key,),
        )
        if row is not None and row["lease_expires_at_ms"] > now_ms:
            if row["owner_id"] != owner_id:
                raise RuntimeLeaseBusyError(
                    runtime_key,
                    row["owner_id"],
                    _ms_to_datetime(row["lease_expires_at_ms"]),
                )
            lease_until_ms = max(row["lease_expires_at_ms"], _datetime_to_ms(lease_until))
            cursor = connection.execute(
                """
                UPDATE runtime_leases
                SET lease_expires_at_ms = ?, heartbeat_at_ms = ?
                WHERE runtime_key = ? AND owner_id = ? AND lease_generation = ?
                  AND lease_expires_at_ms > ?
                """,
                (
                    lease_until_ms,
                    now_ms,
                    runtime_key,
                    owner_id,
                    row["lease_generation"],
                    now_ms,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeLeaseBusyError(
                    runtime_key,
                    row["owner_id"],
                    _ms_to_datetime(row["lease_expires_at_ms"]),
                )
            generation = row["lease_generation"]
            endpoint = row["endpoint"]
        else:
            generation = 1 if row is None else int(row["lease_generation"]) + 1
            lease_until_ms = _datetime_to_ms(lease_until)
            endpoint = None
            connection.execute(
                """
                INSERT INTO runtime_leases (
                  runtime_key, owner_id, lease_generation, endpoint,
                  lease_expires_at_ms, heartbeat_at_ms
                ) VALUES (?, ?, ?, NULL, ?, ?)
                ON CONFLICT(runtime_key) DO UPDATE SET
                  owner_id = excluded.owner_id,
                  lease_generation = excluded.lease_generation,
                  endpoint = NULL,
                  lease_expires_at_ms = excluded.lease_expires_at_ms,
                  heartbeat_at_ms = excluded.heartbeat_at_ms
                """,
                (runtime_key, owner_id, generation, lease_until_ms, now_ms),
            )
        return RuntimeLease(
            runtime_key=runtime_key,
            owner_id=owner_id,
            generation=generation,
            endpoint=endpoint,
            lease_until=_ms_to_datetime(lease_until_ms),
            heartbeat_at=_ms_to_datetime(now_ms),
        )

    return _run_immediate(connection, acquire)


def _renew_runtime_lease_transaction(
    connection: sqlite3.Connection,
    lease: RuntimeLease,
    lease_until: datetime,
) -> bool:
    def renew() -> bool:
        now_ms = _now_ms()
        cursor = connection.execute(
            """
            UPDATE runtime_leases
            SET lease_expires_at_ms = ?, heartbeat_at_ms = ?, endpoint = ?
            WHERE runtime_key = ? AND owner_id = ? AND lease_generation = ?
              AND lease_expires_at_ms > ?
            """,
            (
                _datetime_to_ms(lease_until),
                now_ms,
                lease.endpoint,
                lease.runtime_key,
                lease.owner_id,
                lease.generation,
                now_ms,
            ),
        )
        return cursor.rowcount == 1

    return _run_immediate(connection, renew)


def _release_runtime_lease_transaction(
    connection: sqlite3.Connection,
    lease: RuntimeLease,
) -> None:
    def release() -> None:
        now_ms = _now_ms()
        connection.execute(
            """
            UPDATE runtime_leases
            SET lease_expires_at_ms = 0, heartbeat_at_ms = ?
            WHERE runtime_key = ? AND owner_id = ? AND lease_generation = ?
              AND lease_expires_at_ms > ?
            """,
            (
                now_ms,
                lease.runtime_key,
                lease.owner_id,
                lease.generation,
                now_ms,
            ),
        )

    _run_immediate(connection, release)


def _prune_transaction(
    connection: sqlite3.Connection,
    policy: PrunePolicy,
    now: datetime,
) -> PruneResult:
    def prune() -> PruneResult:
        now_ms = _datetime_to_ms(now)
        terminal_ids: tuple[str, ...] = ()
        if policy.terminal_job_before is not None:
            cutoff_ms = min(_datetime_to_ms(policy.terminal_job_before), now_ms)
            terminal_ids = tuple(
                row[0]
                for row in connection.execute(
                    """
                    SELECT job_id
                    FROM jobs
                    WHERE state IN ('completed', 'failed', 'cancelled')
                      AND terminal_at_ms IS NOT NULL
                      AND terminal_at_ms < ?
                      AND (lease_expires_at_ms IS NULL OR lease_expires_at_ms <= ?)
                      AND NOT EXISTS (
                        SELECT 1 FROM pending_inputs AS p
                        WHERE p.job_id = jobs.job_id AND p.status = 'pending'
                      )
                    """,
                    (cutoff_ms, now_ms),
                ).fetchall()
            )

        events_deleted = 0
        for job_id in terminal_ids:
            events_deleted += _delete_terminal_job(connection, job_id)

        if policy.event_before is not None:
            cutoff_ms = min(_datetime_to_ms(policy.event_before), now_ms)
            events_deleted += _prune_old_events(connection, cutoff_ms)

        return PruneResult(
            terminal_jobs_deleted=len(terminal_ids),
            events_deleted=events_deleted,
            raw_diagnostics_deleted=0,
        )

    return _run_immediate(connection, prune)


def _delete_terminal_job(connection: sqlite3.Connection, job_id: str) -> int:
    events_deleted = int(
        connection.execute(
            """
            SELECT count(*) FROM job_events
            WHERE job_id = ? AND event_type != ?
            """,
            (job_id, _EVENT_WATERMARK_TYPE),
        ).fetchone()[0]
    )
    connection.execute("DELETE FROM job_events WHERE job_id = ?", (job_id,))
    connection.execute("DELETE FROM pending_inputs WHERE job_id = ?", (job_id,))
    connection.execute("DELETE FROM job_results WHERE job_id = ?", (job_id,))
    connection.execute("DELETE FROM job_attempts WHERE job_id = ?", (job_id,))
    connection.execute("DELETE FROM idempotency_keys WHERE job_id = ?", (job_id,))
    connection.execute(
        """
        DELETE FROM provider_references AS doomed
        WHERE doomed.job_id = ?
          AND doomed.session_id IS NOT NULL
              AND EXISTS (
                SELECT 1 FROM provider_references AS retained
                WHERE retained.provider_reference_id != doomed.provider_reference_id
                  AND retained.backend_id = doomed.backend_id
                  AND retained.kind = doomed.kind
                  AND retained.value = doomed.value
                  AND retained.session_id = doomed.session_id
                  AND (
                    (retained.job_id IS NULL AND retained.attempt_number IS NULL)
                    OR (
                      retained.job_id = doomed.job_id
                      AND retained.provider_reference_id < doomed.provider_reference_id
                    )
                  )
              )
        """,
        (job_id,),
    )
    connection.execute(
        """
        UPDATE provider_references
        SET job_id = NULL, attempt_number = NULL
        WHERE job_id = ? AND session_id IS NOT NULL
        """,
        (job_id,),
    )
    connection.execute(
        "DELETE FROM provider_references WHERE job_id = ? AND session_id IS NULL",
        (job_id,),
    )
    cursor = connection.execute(
        """
        DELETE FROM jobs
        WHERE job_id = ?
          AND state IN ('completed', 'failed', 'cancelled')
          AND NOT EXISTS (
            SELECT 1 FROM pending_inputs
            WHERE pending_inputs.job_id = jobs.job_id AND status = 'pending'
          )
        """,
        (job_id,),
    )
    if cursor.rowcount != 1:
        raise StoreSchemaError("eligible terminal job changed during pruning")
    return events_deleted


def _prune_old_events(connection: sqlite3.Connection, cutoff_ms: int) -> int:
    candidates = _fetch_all_mappings(
        connection,
        """
        SELECT job_id, count(*) AS event_count
        FROM job_events
        WHERE event_type != ? AND created_at_ms < ?
        GROUP BY job_id
        """,
        (_EVENT_WATERMARK_TYPE, cutoff_ms),
    )
    deleted = 0
    for candidate in candidates:
        job_id = candidate["job_id"]
        deleted += int(candidate["event_count"])
        maximum = connection.execute(
            """
            SELECT sequence, event_type, created_at_ms
            FROM job_events
            WHERE job_id = ?
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()
        if maximum is None:
            continue
        max_sequence, max_type, max_created_at = maximum
        preserve_maximum = max_type != _EVENT_WATERMARK_TYPE and max_created_at < cutoff_ms
        if preserve_maximum:
            connection.execute(
                """
                DELETE FROM job_events
                WHERE job_id = ? AND event_type != ? AND created_at_ms < ?
                  AND sequence != ?
                """,
                (job_id, _EVENT_WATERMARK_TYPE, cutoff_ms, max_sequence),
            )
            connection.execute(
                """
                UPDATE job_events
                SET event_type = ?, payload_json = '{}', payload_schema_version = ?,
                    attempt_number = NULL, provider_event_type = NULL,
                    provider_event_id = NULL, provider_reference_id = NULL
                WHERE job_id = ? AND sequence = ?
                """,
                (_EVENT_WATERMARK_TYPE, _JSON_SCHEMA_VERSION, job_id, max_sequence),
            )
        else:
            connection.execute(
                """
                DELETE FROM job_events
                WHERE job_id = ? AND event_type != ? AND created_at_ms < ?
                """,
                (job_id, _EVENT_WATERMARK_TYPE, cutoff_ms),
            )
    return deleted


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


def _insert_job_event(
    connection: sqlite3.Connection,
    job_id: str,
    event: BackendEvent,
    attempt_number: int | None,
    now_ms: int,
) -> JobEvent:
    row = connection.execute(
        "SELECT backend_id FROM jobs WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        raise JobNotFoundError(job_id)
    sequence = int(
        connection.execute(
            "SELECT coalesce(max(sequence), 0) + 1 FROM job_events WHERE job_id = ?",
            (job_id,),
        ).fetchone()[0]
    )
    provider_reference_id = _ensure_job_provider_reference(
        connection,
        job_id,
        row[0],
        event.provider_reference,
        now_ms,
    )
    connection.execute(
        """
        INSERT INTO job_events (
          job_id, sequence, event_type, payload_json, payload_schema_version,
          attempt_number, created_at_ms, provider_event_type, provider_event_id,
          provider_reference_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
        """,
        (
            job_id,
            sequence,
            event.type,
            _canonical_json(dict(event.payload)),
            _JSON_SCHEMA_VERSION,
            attempt_number,
            _datetime_to_ms(event.occurred_at),
            event.provider_event_type,
            provider_reference_id,
        ),
    )
    event_row = _fetch_one_mapping(
        connection,
        "SELECT * FROM job_events WHERE job_id = ? AND sequence = ?",
        (job_id, sequence),
    )
    if event_row is None:
        raise StoreSchemaError("inserted job event is missing")
    return _event_from_row(connection, event_row)


def _read_job_result(
    connection: sqlite3.Connection,
    job_id: str,
) -> JobResultEnvelope | JobError | None:
    row = _fetch_one_mapping(
        connection,
        """
        SELECT j.state AS job_state,
               r.job_id AS result_job_id,
               r.outcome_kind,
               r.payload_json,
               r.payload_schema_version,
               r.error_json,
               r.error_schema_version,
               r.created_at_ms
        FROM jobs AS j
        LEFT JOIN job_results AS r ON r.job_id = j.job_id
        WHERE j.job_id = ?
        """,
        (job_id,),
    )
    if row is None:
        return None
    job_state = row["job_state"]
    has_result = row["result_job_id"] is not None
    if job_state not in TERMINAL_STATES:
        if has_result:
            raise StoreSchemaError("nonterminal job has a terminal result row")
        return None
    if not has_result:
        raise StoreSchemaError("terminal job is missing its result row")

    outcome_kind = row["outcome_kind"]
    payload_json = row["payload_json"]
    payload_version = row["payload_schema_version"]
    error_json = row["error_json"]
    error_version = row["error_schema_version"]
    if outcome_kind == "failed":
        if (
            job_state != "failed"
            or error_json is None
            or error_version is None
            or payload_json is not None
            or payload_version is not None
        ):
            raise StoreSchemaError("failed job result row has an invalid state or shape")
        return _decode_model(
            error_json,
            error_version,
            "error_schema_version",
            JobError,
        )
    if outcome_kind == "succeeded":
        if (
            job_state != "completed"
            or payload_json is None
            or payload_version is None
            or error_json is not None
            or error_version is not None
        ):
            raise StoreSchemaError("succeeded job result row has an invalid state or shape")
        _require_json_version(payload_version, "payload_schema_version")
        try:
            payload = _OPERATION_RESULT_ADAPTER.validate_json(payload_json)
        except (ValidationError, ValueError) as error:
            raise StoreSchemaError("job result payload is invalid") from error
        return JobResultEnvelope(
            job_id=job_id,
            payload=payload,
            completed_at=_ms_to_datetime(row["created_at_ms"]),
        )
    if outcome_kind == "cancelled":
        if (
            job_state != "cancelled"
            or payload_json is not None
            or payload_version is not None
            or error_json is not None
            or error_version is not None
        ):
            raise StoreSchemaError("cancelled job result row has an invalid state or shape")
        return None
    raise StoreSchemaError(f"unknown job result outcome: {outcome_kind}")


def _read_job_attempts(
    connection: sqlite3.Connection,
    job_id: str,
) -> tuple[JobAttempt, ...]:
    rows = _fetch_all_mappings(
        connection,
        "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY attempt_number",
        (job_id,),
    )
    attempts: list[JobAttempt] = []
    for row in rows:
        error = None
        if row["error_json"] is not None:
            error = _decode_model(
                row["error_json"],
                row["error_schema_version"],
                "error_schema_version",
                JobError,
            )
        elif row["error_schema_version"] is not None:
            raise StoreSchemaError("attempt error schema version has no payload")
        attempts.append(
            JobAttempt(
                job_id=row["job_id"],
                attempt_number=row["attempt_number"],
                phase=cast("JobPhase", row["phase"]),
                worker_id=row["owner_id"],
                lease_generation=row["lease_generation"],
                lease_expires_at=_optional_ms(row["lease_expires_at_ms"]),
                heartbeat_at=_optional_ms(row["heartbeat_at_ms"]),
                retry_classification=row["retry_classification"],
                reconciliation_classification=row["reconciliation_classification"],
                started_at=_ms_to_datetime(row["started_at_ms"]),
                ended_at=_optional_ms(row["ended_at_ms"]),
                error_code=None if error is None else error.code,
                error_message=None if error is None else error.message,
            )
        )
    return tuple(attempts)


def _read_pending_inputs(
    connection: sqlite3.Connection,
    job_id: str,
    *,
    unresolved_only: bool,
) -> tuple[PendingInput, ...]:
    predicate = " AND status = 'pending'" if unresolved_only else ""
    rows = _fetch_all_mappings(
        connection,
        f"""
        SELECT * FROM pending_inputs
        WHERE job_id = ?{predicate}
        ORDER BY created_at_ms, input_id
        """,
        (job_id,),
    )
    return tuple(_pending_input_from_row(connection, row) for row in rows)


def _pending_input_from_row(
    connection: sqlite3.Connection,
    row: dict[str, Any],
) -> PendingInput:
    _require_json_version(row["request_schema_version"], "request_schema_version")
    try:
        request = _INPUT_REQUEST_ADAPTER.validate_json(row["request_json"])
    except (ValidationError, ValueError) as error:
        raise StoreSchemaError("pending input request is invalid") from error
    if request.kind != row["kind"]:
        raise StoreSchemaError("pending input kind does not match its request")
    response = None
    if row["response_json"] is not None:
        _require_json_version(row["response_schema_version"], "response_schema_version")
        try:
            response = _INPUT_RESPONSE_ADAPTER.validate_json(row["response_json"])
        except (ValidationError, ValueError) as error:
            raise StoreSchemaError("pending input response is invalid") from error
    elif row["response_schema_version"] is not None:
        raise StoreSchemaError("pending input response schema version has no payload")
    if (row["status"] == "pending") != (response is None):
        raise StoreSchemaError("pending input status does not match its response")
    provider_reference = _read_scoped_provider_reference(
        connection,
        row["provider_reference_id"],
        row["job_id"],
    )
    try:
        return PendingInput(
            input_id=row["input_id"],
            job_id=row["job_id"],
            request=request,
            provider_reference=provider_reference,
            created_at=_ms_to_datetime(row["created_at_ms"]),
            resolved_at=_optional_ms(row["resolved_at_ms"]),
            response=response,
        )
    except ValidationError as error:
        raise StoreSchemaError("pending input row is invalid") from error


def _read_scoped_provider_reference(
    connection: sqlite3.Connection,
    provider_reference_id: str | None,
    job_id: str,
) -> ProviderReference | None:
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
    if row is None or row[2] != job_id:
        raise StoreSchemaError("provider reference is missing or belongs to another job")
    try:
        return ProviderReference(kind=row[0], value=row[1])
    except ValidationError as error:
        raise StoreSchemaError("provider reference row is invalid") from error


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
        WHERE job_id = ? AND sequence > ? AND event_type != ?
        ORDER BY sequence
        LIMIT ?
        """,
        (job_id, after_sequence, _EVENT_WATERMARK_TYPE, limit + 1),
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
