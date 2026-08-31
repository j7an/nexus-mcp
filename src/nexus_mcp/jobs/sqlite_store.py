"""Dedicated-thread SQLite lifecycle and schema foundation for durable jobs."""

import asyncio
import os
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

from nexus_mcp.jobs.migrations import MIGRATIONS, Migration
from nexus_mcp.jobs.paths import default_database_path

__all__ = ["SQLiteJobStore"]

_ResultT = TypeVar("_ResultT")

_CONNECTION_PRAGMAS = (
    "PRAGMA journal_mode = WAL;",
    "PRAGMA foreign_keys = ON;",
    "PRAGMA synchronous = FULL;",
    "PRAGMA busy_timeout = 5000;",
)


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
        """Run one connection operation on the dedicated worker thread."""
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
        return result

    def _close_in_worker(self) -> None:
        if self._connection is None:
            return
        self._connection.close()
        self._connection = None


class SQLiteJobStore:
    """SQLite job-store lifecycle whose domain operations arrive in the next task."""

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
