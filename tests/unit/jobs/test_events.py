"""Durable event-subscription behavior over the job-store contract."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from nexus_mcp.jobs.events import EventNotifier, EventPollingPolicy, JobEventSubscription
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore
from nexus_mcp.jobs.store import ClaimedJob, JobStore
from tests.unit.jobs._store_contract_support import (
    make_cancel_job_command,
    make_create_job_command,
    make_event,
)

NOW = datetime(2026, 8, 30, 20, 0, tzinfo=UTC)
LEASE_UNTIL = datetime(2099, 1, 1, tzinfo=UTC)


@pytest.fixture
async def job_store(tmp_path: Path):
    """Open a private SQLite store for subscription behavior."""
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    await store.open()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
async def terminal_job(job_store: SQLiteJobStore):
    """Create a terminal job with three committed, ordered events."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next(
        "worker-1", LEASE_UNTIL, event=make_event("provider_connected")
    )
    assert claimed is not None
    await job_store.request_cancel(make_cancel_job_command(created.handle.job_id))
    terminal = await job_store.get_job(created.handle.job_id)
    assert terminal is not None and terminal.state == "cancelled"
    return terminal


@pytest.fixture
async def active_job(job_store: SQLiteJobStore) -> ClaimedJob:
    """Create a claimed job whose current sequence is two."""
    await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next(
        "worker-1", LEASE_UNTIL, event=make_event("provider_connected")
    )
    assert claimed is not None
    return claimed


async def append_and_notify(
    job_store: SQLiteJobStore, notifier: EventNotifier, active_job: ClaimedJob
) -> None:
    """Commit one progress event before waking local subscribers."""
    await job_store.append_events(active_job.token, (make_event("progress"),))
    notifier.notify()


class _ObservedStore:
    """Expose when a real store read has found no later durable events."""

    def __init__(self, store: SQLiteJobStore) -> None:
        self._store = store
        self.empty_page_read = asyncio.Event()

    async def read_events(self, job_id: str, after_sequence: int, limit: int):
        """Delegate the durable page read and record an empty result."""
        page = await self._store.read_events(job_id, after_sequence, limit)
        if not page.events:
            self.empty_page_read.set()
        return page

    async def get_job(self, job_id: str):
        """Delegate the terminal snapshot read used by the subscription."""
        return await self._store.get_job(job_id)


async def test_subscription_resumes_after_cursor(job_store, terminal_job):
    """Removing cursor resume or terminal draining would omit committed history."""
    subscription = JobEventSubscription(job_store, EventNotifier(), terminal_job.job_id, after=1)

    events = [event async for event in subscription]

    assert [event.sequence for event in events] == [2, 3]


async def test_local_commit_wakes_without_waiting_for_poll(job_store, active_job):
    """Dropping the notifier wake would leave the reader asleep for the five-second poll."""
    notifier = EventNotifier()
    observed_store = _ObservedStore(job_store)
    subscription = JobEventSubscription(
        cast("JobStore", observed_store),
        notifier,
        active_job.token.job_id,
        after=2,
        polling_policy=EventPollingPolicy(minimum_seconds=5, maximum_seconds=5),
    )
    waiter = asyncio.create_task(anext(subscription))
    async with asyncio.timeout(0.5):
        await observed_store.empty_page_read.wait()

    await append_and_notify(job_store, notifier, active_job)

    async with asyncio.timeout(0.5):
        event = await waiter
    assert event.event_type == "progress"
