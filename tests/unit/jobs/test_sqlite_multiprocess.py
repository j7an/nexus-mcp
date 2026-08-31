"""Real multi-process SQLite claim and generation-fence regressions."""

import asyncio
import multiprocessing
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from typing import Any

from nexus_mcp.core import BackendEvent, StaleLeaseError
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore
from tests.unit.jobs.test_store_contract import make_create_job_command

OLD = datetime(2025, 1, 1, tzinfo=UTC)
LEASE_UNTIL = datetime(2099, 1, 1, tzinfo=UTC)


def _event(event_type: str) -> BackendEvent:
    return BackendEvent(type=event_type, occurred_at=OLD)


async def _claim_once(
    database_path: str,
    barrier: Any,
    results: Any,
    owner_id: str,
) -> None:
    store = SQLiteJobStore(Path(database_path))
    await store.open()
    try:
        barrier.wait()
        claimed = await store.claim_next(owner_id, LEASE_UNTIL, event=_event("progress"))
        results.put(
            (
                owner_id,
                None if claimed is None else claimed.job.job_id,
                None if claimed is None else claimed.token.generation,
            )
        )
    finally:
        await store.close()


def _claim_once_entry(database_path: str, barrier: Any, results: Any, owner_id: str) -> None:
    """Spawn-safe entrypoint for one contended claim."""
    asyncio.run(_claim_once(database_path, barrier, results, owner_id))


async def _claim_then_write_stale(
    database_path: str,
    claimed_barrier: Any,
    reclaimed_barrier: Any,
    results: Any,
) -> None:
    store = SQLiteJobStore(Path(database_path))
    await store.open()
    try:
        claimed = await store.claim_next("worker-old", OLD, event=_event("progress"))
        assert claimed is not None
        results.put(("old_generation", claimed.token.generation))
        claimed_barrier.wait()
        reclaimed_barrier.wait()
        try:
            await store.append_events(claimed.token, (_event("message"),))
        except StaleLeaseError:
            results.put(("stale_write", True))
        else:
            results.put(("stale_write", False))
    finally:
        await store.close()


def _claim_then_write_stale_entry(
    database_path: str,
    claimed_barrier: Any,
    reclaimed_barrier: Any,
    results: Any,
) -> None:
    """Spawn-safe entrypoint that attempts a write after another process reclaims."""
    asyncio.run(_claim_then_write_stale(database_path, claimed_barrier, reclaimed_barrier, results))


async def _reclaim_after_barrier(
    database_path: str,
    claimed_barrier: Any,
    reclaimed_barrier: Any,
    results: Any,
) -> None:
    store = SQLiteJobStore(Path(database_path))
    await store.open()
    try:
        claimed_barrier.wait()
        reclaimed = await store.claim_next(
            "worker-new", LEASE_UNTIL, event=_event("reconciliation")
        )
        assert reclaimed is not None
        results.put(("new_generation", reclaimed.token.generation))
        reclaimed_barrier.wait()
    finally:
        await store.close()


def _reclaim_after_barrier_entry(
    database_path: str,
    claimed_barrier: Any,
    reclaimed_barrier: Any,
    results: Any,
) -> None:
    """Spawn-safe entrypoint for a deterministic expired-lease reclaim."""
    asyncio.run(_reclaim_after_barrier(database_path, claimed_barrier, reclaimed_barrier, results))


def _join_processes(processes: tuple[multiprocessing.Process, ...]) -> None:
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert [(process.pid, process.exitcode) for process in processes] == [
        (process.pid, 0) for process in processes
    ]


def _drain_results(results: Any, expected: int) -> list[tuple[Any, ...]]:
    collected: list[tuple[Any, ...]] = []
    for _ in range(expected):
        try:
            collected.append(results.get(timeout=5))
        except Empty as error:
            raise AssertionError(f"expected {expected} child results, got {collected}") from error
    return collected


async def test_spawned_claimers_commit_exactly_one_claim(tmp_path: Path):
    """BEGIN IMMEDIATE serializes two ready processes so only one gets the queued job."""
    database_path = tmp_path / "claims.sqlite3"
    store = SQLiteJobStore(database_path)
    await store.open()
    created = await store.create_job(make_create_job_command())
    await store.close()

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = tuple(
        context.Process(
            target=_claim_once_entry,
            args=(str(database_path), barrier, results, owner_id),
        )
        for owner_id in ("worker-a", "worker-b")
    )
    for process in processes:
        process.start()
    _join_processes(processes)

    claims = _drain_results(results, 2)
    winners = [claim for claim in claims if claim[1] is not None]
    assert len(winners) == 1
    assert winners[0][1:] == (created.handle.job_id, 1)


async def test_spawned_reclaim_fences_former_owner_write(tmp_path: Path):
    """A cross-process reclaim rejects the former generation's event write."""
    database_path = tmp_path / "reclaim.sqlite3"
    store = SQLiteJobStore(database_path)
    await store.open()
    await store.create_job(make_create_job_command())
    await store.close()

    context = multiprocessing.get_context("spawn")
    claimed_barrier = context.Barrier(2)
    reclaimed_barrier = context.Barrier(2)
    results = context.Queue()
    processes = (
        context.Process(
            target=_claim_then_write_stale_entry,
            args=(str(database_path), claimed_barrier, reclaimed_barrier, results),
        ),
        context.Process(
            target=_reclaim_after_barrier_entry,
            args=(str(database_path), claimed_barrier, reclaimed_barrier, results),
        ),
    )
    for process in processes:
        process.start()
    _join_processes(processes)

    observed = dict(_drain_results(results, 3))
    assert observed == {"old_generation": 1, "new_generation": 2, "stale_write": True}
