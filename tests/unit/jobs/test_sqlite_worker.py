"""Dedicated-thread and connection-policy behavior for SQLite."""

import sqlite3
import threading

import pytest

from nexus_mcp.jobs import sqlite_store
from nexus_mcp.jobs.sqlite_store import _SQLiteWorker


async def test_worker_creates_and_uses_connection_on_one_dedicated_thread(tmp_path):
    """Connection creation and all operations stay off the event-loop thread."""
    worker = _SQLiteWorker(tmp_path / "worker.sqlite3")
    event_loop_thread = threading.get_ident()

    await worker.open()
    try:
        first_thread = await worker._call(lambda _connection: threading.get_ident())
        second_thread = await worker._call(lambda _connection: threading.get_ident())
    finally:
        await worker.close()

    assert first_thread == second_thread
    assert first_thread != event_loop_thread


async def test_worker_keeps_default_same_thread_connection_guard(tmp_path):
    """SQLite itself rejects use of the worker's connection from another thread."""
    worker = _SQLiteWorker(tmp_path / "guarded.sqlite3")

    def attempt_cross_thread_use(connection: sqlite3.Connection) -> BaseException | None:
        errors: list[BaseException] = []

        def use_connection() -> None:
            try:
                connection.execute("SELECT 1")
            except BaseException as error:  # noqa: BLE001 - the exact SQLite error is asserted.
                errors.append(error)

        thread = threading.Thread(target=use_connection)
        thread.start()
        thread.join()
        return errors[0] if errors else None

    await worker.open()
    try:
        error = await worker._call(attempt_cross_thread_use)
    finally:
        await worker.close()

    assert isinstance(error, sqlite3.ProgrammingError)


async def test_worker_applies_required_connection_pragmas(tmp_path):
    """Every worker connection uses the durable WAL and foreign-key policy."""
    worker = _SQLiteWorker(tmp_path / "pragmas.sqlite3")

    def read_pragmas(connection: sqlite3.Connection) -> tuple[str, int, int, int]:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        return journal_mode, foreign_keys, synchronous, busy_timeout

    await worker.open()
    try:
        assert await worker._call(read_pragmas) == ("wal", 1, 2, 5000)
    finally:
        await worker.close()


async def test_worker_prevents_connection_escape(tmp_path):
    """An operation cannot return the owned connection to another module."""
    worker = _SQLiteWorker(tmp_path / "private-connection.sqlite3")

    await worker.open()
    try:
        with pytest.raises(RuntimeError, match="connection cannot escape"):
            await worker._call(lambda connection: connection)
    finally:
        await worker.close()


async def test_worker_prevents_cursor_escape(tmp_path):
    """An operation cannot return a cursor that retains the owned connection."""
    worker = _SQLiteWorker(tmp_path / "private-cursor.sqlite3")

    await worker.open()
    try:
        with pytest.raises(RuntimeError, match="cursor cannot escape"):
            await worker._call(lambda connection: connection.execute("SELECT 1"))
    finally:
        await worker.close()


async def test_worker_close_is_idempotent_and_final(tmp_path):
    """Repeated close is safe and no work can run after executor shutdown."""
    worker = _SQLiteWorker(tmp_path / "closed.sqlite3")

    await worker.open()
    await worker.close()
    await worker.close()

    with pytest.raises(RuntimeError, match="closed"):
        await worker._call(lambda connection: connection.execute("SELECT 1").fetchone())


def test_pragma_non_busy_error_raises_without_retry(tmp_path):
    """Only SQLITE_BUSY is retried; any other SQLite error propagates at once."""
    connection = sqlite3.connect(tmp_path / "error.sqlite3")
    try:
        with pytest.raises(sqlite3.OperationalError) as raised:
            sqlite_store._execute_pragma(connection, "NOT A PRAGMA;")
    finally:
        connection.close()

    assert raised.value.sqlite_errorcode != sqlite3.SQLITE_BUSY


def test_pragma_busy_retry_gives_up_at_deadline(tmp_path, monkeypatch):
    """A lock that outlives the deadline surfaces SQLITE_BUSY instead of spinning."""
    database_path = tmp_path / "busy.sqlite3"
    monkeypatch.setattr(sqlite_store, "_PRAGMA_BUSY_DEADLINE_SECONDS", 0.0)
    holder = sqlite3.connect(database_path, isolation_level=None)
    waiter = sqlite3.connect(database_path, isolation_level=None, timeout=0)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError) as raised:
            sqlite_store._execute_pragma(waiter, "PRAGMA journal_mode = WAL;")
    finally:
        waiter.close()
        holder.close()

    assert raised.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
