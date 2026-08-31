"""Unit contracts for fence-bound worker control and output handling."""

import unicodedata

import pytest
from pydantic import ValidationError

from nexus_mcp.core import BackendEvent, RetryPolicy
from nexus_mcp.jobs.control import OutputChunker
from nexus_mcp.jobs.store import ControlSnapshot
from nexus_mcp.jobs.worker import ExponentialRetryDelay, WorkerPolicy
from tests.fixtures import make_pending_permission


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
