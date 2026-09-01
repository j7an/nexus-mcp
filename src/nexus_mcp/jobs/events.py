"""Durable job-event subscriptions backed by committed store history."""

import asyncio
from collections import deque
from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus_mcp.core import TERMINAL_STATES, JobEvent
from nexus_mcp.jobs.store import JobStore

__all__ = [
    "EventNotifier",
    "EventPollingPolicy",
    "JobEventSubscription",
    "subscribe_events",
]


class EventPollingPolicy(BaseModel):
    """Bounded adaptive polling cadence for durable event reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_seconds: float = Field(default=0.05, gt=0)
    maximum_seconds: float = Field(default=0.5, gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "EventPollingPolicy":
        """Reject a delay range whose lower bound exceeds its upper bound."""
        if self.minimum_seconds > self.maximum_seconds:
            raise ValueError("minimum_seconds must be less than or equal to maximum_seconds")
        return self


class EventNotifier:
    """Process-local wake signal that never replaces durable polling."""

    def __init__(self) -> None:
        self._changed = asyncio.Event()
        self._revision = 0

    @property
    def revision(self) -> int:
        """Return the current in-process notification revision."""
        return self._revision

    def notify(self) -> None:
        """Wake local readers after their corresponding event transaction commits."""
        self._revision += 1
        self._changed.set()

    async def wait_for_change(self, revision: int) -> None:
        """Wait until a notification newer than ``revision`` is observed."""
        while self._revision == revision:
            self._changed.clear()
            if self._revision != revision:
                return
            await self._changed.wait()


class JobEventSubscription(AsyncIterator[JobEvent]):
    """Yield one job's committed events in sequence order after a durable cursor."""

    def __init__(
        self,
        store: JobStore,
        notifier: EventNotifier,
        job_id: str,
        after: int = 0,
        *,
        polling_policy: EventPollingPolicy | None = None,
        page_size: int = 100,
    ) -> None:
        if after < 0:
            raise ValueError("after must be non-negative")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be from 1 through 1000")
        self._store = store
        self._notifier = notifier
        self._job_id = job_id
        self._after = after
        self._policy = polling_policy or EventPollingPolicy()
        self._page_size = page_size
        self._delay_seconds = self._policy.minimum_seconds
        self._pending: deque[JobEvent] = deque()
        self._stopped = False

    def __aiter__(self) -> "JobEventSubscription":
        """Return this one-pass event stream."""
        return self

    async def __anext__(self) -> JobEvent:
        """Return the next committed event or stop after terminal history drains."""
        if self._stopped:
            raise StopAsyncIteration
        if self._pending:
            return self._next_pending()

        while True:
            notifier_revision = self._notifier.revision
            page = await self._store.read_events(self._job_id, self._after, self._page_size)
            if page.events:
                self._pending.extend(page.events)
                self._delay_seconds = self._policy.minimum_seconds
                return self._next_pending()

            job = await self._store.get_job(self._job_id)
            if job is None or job.state in TERMINAL_STATES:
                final_page = await self._store.read_events(
                    self._job_id, self._after, self._page_size
                )
                if final_page.events:
                    self._pending.extend(final_page.events)
                    self._delay_seconds = self._policy.minimum_seconds
                    return self._next_pending()
                self._stopped = True
                raise StopAsyncIteration

            await self._wait_for_update(notifier_revision, self._delay_seconds)
            self._delay_seconds = min(self._delay_seconds * 2, self._policy.maximum_seconds)

    def _next_pending(self) -> JobEvent:
        """Advance the durable cursor only for the event returned to the consumer."""
        event = self._pending.popleft()
        self._after = event.sequence
        return event

    async def _wait_for_update(self, revision: int, timeout_seconds: float) -> None:
        """Wake on local notification or poll deadline and cancel every helper task."""
        notifier_task = asyncio.create_task(self._notifier.wait_for_change(revision))
        timeout_task = asyncio.create_task(asyncio.sleep(timeout_seconds))
        tasks = (notifier_task, timeout_task)
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def subscribe_events(
    store: JobStore,
    notifier: EventNotifier,
    job_id: str,
    after: int = 0,
    *,
    polling_policy: EventPollingPolicy | None = None,
    page_size: int = 100,
) -> JobEventSubscription:
    """Create a durable event stream that resumes strictly after ``after``."""
    return JobEventSubscription(
        store,
        notifier,
        job_id,
        after,
        polling_policy=polling_policy,
        page_size=page_size,
    )
