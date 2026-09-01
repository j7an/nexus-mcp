"""Unit contracts for fence-bound worker control and output handling."""

import unicodedata
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from nexus_mcp.core import (
    BackendEvent,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    RetryPolicy,
)
from nexus_mcp.jobs.control import OutputChunker, StoreBackedExecutionContext
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.store import ControlSnapshot, CreateJobCommand
from nexus_mcp.jobs.worker import ExponentialRetryDelay, WorkerPolicy
from tests.fixtures import make_pending_permission, make_workspace
from tests.job_fakes import InMemoryJobStore


def test_worker_policy_requires_two_heartbeats_inside_every_lease():
    """A lease policy cannot leave only one heartbeat opportunity before expiry."""
    with pytest.raises(ValidationError):
        WorkerPolicy(lease_seconds=20, heartbeat_seconds=10)

    policy = WorkerPolicy(lease_seconds=20.1, heartbeat_seconds=10)
    assert policy.heartbeat_seconds == 10


def test_exponential_retry_delay_uses_bounded_full_jitter_and_retry_after():
    """Retry scheduling honors both the exponential cap and provider minimum delay."""
    delay = ExponentialRetryDelay(random_value=lambda: 0.5)
    policy = RetryPolicy(max_attempts=4, base_delay_seconds=2, max_delay_seconds=5)

    assert delay(attempt=3, retry_after=None, policy=policy) == 2.5
    assert delay(attempt=3, retry_after=4.0, policy=policy) == 4.0


def test_control_snapshot_rejects_unresolved_records_in_resolved_partition():
    """A worker must never mistake a merely pending request for a durable response."""
    with pytest.raises(ValidationError):
        ControlSnapshot(
            state="input_required",
            cancel_requested=False,
            lease_generation=1,
            resolved_inputs=(make_pending_permission(),),
        )


async def test_output_chunker_coalesces_normalized_utf8_within_byte_bound():
    """Small provider deltas become fewer valid normalized message journal rows."""
    emitted: list[BackendEvent] = []

    async def emit(event: BackendEvent) -> None:
        emitted.append(event)

    chunker = OutputChunker(emit, max_bytes=8)
    deltas = ("e", "\u0301", "-", "🙂", "-", "alpha", "\ud800", "-", "omega")
    for delta in deltas:
        await chunker.add(delta)
    await chunker.flush()

    chunks = [str(event.payload["text"]) for event in emitted]
    expected = (
        unicodedata.normalize("NFC", "".join(deltas))
        .encode("utf-8", errors="replace")
        .decode("utf-8")
    )
    assert "".join(chunks) == expected
    assert all(event.type == "message" for event in emitted)
    assert all(len(chunk.encode("utf-8")) <= 8 for chunk in chunks)
    assert len(chunks) < len(deltas)


async def test_output_chunker_flushes_before_a_nonmessage_boundary():
    """Buffered prose cannot be reordered behind a command or terminal boundary."""
    emitted: list[BackendEvent] = []

    async def emit(event: BackendEvent) -> None:
        emitted.append(event)

    chunker = OutputChunker(emit, max_bytes=64)
    await chunker.add("before")
    await chunker.flush()
    await emit(BackendEvent(type="command", payload={"command": "pwd"}))

    assert [event.type for event in emitted] == ["message", "command"]


async def test_store_context_persists_adapter_progress_with_bounded_sanitization():
    """Adapter progress values survive the fence while opaque and oversized data do not."""
    store = InMemoryJobStore()
    workspace = make_workspace()
    created = await store.create_job(
        CreateJobCommand(
            workspace=workspace,
            backend_id="codex",
            owner_id="local:control-test",
            access_policy="private",
            operation={"kind": "turn", "prompt": "Inspect"},
            requested_config=RequestedExecutionConfig(),
            session_id="session-control",
            create_session=True,
            command_family="control-test",
            queued_event=BackendEvent(type="job_queued", payload={"state": "queued"}),
        )
    )
    claimed = await store.claim_next(
        "worker-control",
        datetime.now(UTC) + timedelta(minutes=1),
        event=BackendEvent(type="job_started", payload={"state": "running"}),
    )
    assert claimed is not None
    context = StoreBackedExecutionContext(
        store=store,
        notifier=EventNotifier(),
        token=claimed.token,
        job=claimed.job,
        attempt=claimed.attempt,
        workspace=workspace,
        resolved_config=ResolvedExecutionConfig(),
    )

    await context.emit(
        BackendEvent(
            type="progress",
            payload={
                "progress": 2.0,
                "total": 5.0,
                "message": "x" * 5000,
                "raw_payload": "drop",
            },
        )
    )

    events = await store.read_events(created.handle.job_id, 0, 100)
    progress = [event for event in events.events if event.type == "progress"][-1]
    assert dict(progress.payload) == {
        "progress": 2.0,
        "total": 5.0,
        "message": "x" * 4096,
    }
