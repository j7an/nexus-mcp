"""Fence-bound backend callbacks and normalized durable output chunking."""

import asyncio
import unicodedata
from collections.abc import Awaitable, Callable
from contextlib import suppress

from pydantic import JsonValue

from nexus_mcp.backends import (
    CancelRequested,
    ControlSignal,
    InputResolved,
    LeaseLost,
    RuntimeShutdown,
)
from nexus_mcp.core import (
    AgentJob,
    BackendEvent,
    InputRequest,
    InputResponse,
    JobAttempt,
    PendingInput,
    ProviderReference,
    ResolvedExecutionConfig,
    StaleLeaseError,
    Workspace,
    new_id,
)
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.store import JobStore, LeaseToken

__all__ = ["OutputChunker", "StoreBackedExecutionContext"]

type EventEmitter = Callable[[BackendEvent], Awaitable[None]]

_PAYLOAD_FIELDS: dict[str, frozenset[str]] = {
    "progress": frozenset({"message", "stage", "status", "percent", "current", "total"}),
    "provider_connected": frozenset({"backend_id", "status", "version"}),
    "provider_disconnected": frozenset({"backend_id", "status"}),
    "log": frozenset({"level", "message", "category"}),
    "command": frozenset({"command", "exit_code", "output_summary"}),
    "file_change": frozenset({"path", "status"}),
    "message": frozenset({"text", "final"}),
    "usage": frozenset(
        {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "cost_usd",
        }
    ),
    "input_required": frozenset({"input_id", "kind"}),
    "input_resolved": frozenset({"input_id"}),
}
_REASONING_METADATA_FIELDS = frozenset({"stage", "status", "percent", "current", "total"})


class OutputChunker:
    """Coalesce provider message deltas into bounded, normalized UTF-8 events."""

    def __init__(self, emit: EventEmitter, *, max_bytes: int = 4096) -> None:
        if max_bytes < 4:
            raise ValueError("max_bytes must be at least 4 for one UTF-8 code point")
        self._emit = emit
        self._max_bytes = max_bytes
        self._buffer = ""
        self._lock = asyncio.Lock()

    async def add(self, text: str) -> None:
        """Append one provider delta and emit every complete bounded chunk."""
        if not text:
            return
        async with self._lock:
            self._buffer = _normalize_utf8(self._buffer + text)
            while len(self._buffer.encode("utf-8")) >= self._max_bytes:
                chunk, self._buffer = _split_utf8_prefix(self._buffer, self._max_bytes)
                await self._emit(BackendEvent(type="message", payload={"text": chunk}))

    async def flush(self) -> None:
        """Emit the remaining normalized message text at one semantic boundary."""
        async with self._lock:
            if not self._buffer:
                return
            chunk = _normalize_utf8(self._buffer)
            self._buffer = ""
            await self._emit(BackendEvent(type="message", payload={"text": chunk}))


class StoreBackedExecutionContext:
    """Expose only generation-fenced, durable backend execution effects."""

    def __init__(
        self,
        *,
        store: JobStore,
        notifier: EventNotifier,
        token: LeaseToken,
        job: AgentJob,
        attempt: JobAttempt,
        workspace: Workspace,
        resolved_config: ResolvedExecutionConfig,
        control_poll_seconds: float = 0.25,
        output_chunk_bytes: int = 4096,
    ) -> None:
        if control_poll_seconds <= 0:
            raise ValueError("control_poll_seconds must be positive")
        self.job = job
        self.attempt = attempt
        self.workspace = workspace
        self.resolved_config = resolved_config
        self._store = store
        self._notifier = notifier
        self._token = token
        self._control_poll_seconds = control_poll_seconds
        self._active = True
        self._detached_signal: LeaseLost | RuntimeShutdown | None = None
        self._detached = asyncio.Event()
        self._detach_delivered = asyncio.Event()
        self._seen_resolved_input_ids: set[str] = set()
        self._cancel_observed = False
        self._chunker = OutputChunker(self._append_event, max_bytes=output_chunk_bytes)

    @property
    def token(self) -> LeaseToken:
        """Return the immutable generation fence closed over by every callback."""
        return self._token

    @property
    def cancel_observed(self) -> bool:
        """Return whether durable cancellation was delivered to this context."""
        return self._cancel_observed

    @property
    def detached_signal(self) -> LeaseLost | RuntimeShutdown | None:
        """Return the non-authoritative detach reason when this observation lost ownership."""
        return self._detached_signal

    async def emit(self, event: BackendEvent) -> None:
        """Persist one sanitized provider event under the current generation fence."""
        self._ensure_active()
        await self._chunker.flush()
        await self._append_event(_sanitize_event(event))

    async def emit_output_delta(self, text: str) -> None:
        """Coalesce one provider message delta under the current generation fence."""
        self._ensure_active()
        await self._chunker.add(text)

    async def record_provider_reference(self, reference: ProviderReference) -> None:
        """Persist one provider identity under the current generation fence."""
        self._ensure_active()
        await self._store.record_provider_reference(self._token, reference)

    async def request_input(self, request: InputRequest) -> InputResponse:
        """Persist a request and return only its matching durable cross-task response."""
        self._ensure_active()
        await self._chunker.flush()
        pending = PendingInput(input_id=new_id(), job_id=self.job.job_id, request=request)
        await self._store.mark_input_required(
            self._token,
            (pending,),
            event=BackendEvent(
                type="input_required",
                payload={"input_id": pending.input_id, "kind": request.kind},
            ),
        )
        self._notifier.notify()
        while True:
            snapshot = await self._store.get_control_snapshot(self._token)
            if snapshot.cancel_requested:
                self._cancel_observed = True
                raise asyncio.CancelledError
            for resolved in snapshot.resolved_inputs:
                if resolved.input_id != pending.input_id:
                    continue
                response = resolved.response
                assert response is not None
                if snapshot.state == "input_required" and not snapshot.unresolved_inputs:
                    await self._store.mark_running(
                        self._token,
                        tuple(item.input_id for item in snapshot.resolved_inputs),
                        event=BackendEvent(
                            type="job_started",
                            payload={"resumed": True},
                        ),
                    )
                    self._notifier.notify()
                return response
            self._ensure_active()
            await self._wait_for_update()

    async def wait_for_control(self) -> ControlSignal:
        """Read persisted cancellation and input responses, using notifications only as hints."""
        while True:
            if self._detached_signal is not None:
                self._detach_delivered.set()
                return self._detached_signal
            self._ensure_active()
            snapshot = await self._store.get_control_snapshot(self._token)
            if snapshot.cancel_requested:
                self._cancel_observed = True
                return CancelRequested()
            for resolved in snapshot.resolved_inputs:
                if resolved.input_id in self._seen_resolved_input_ids:
                    continue
                self._seen_resolved_input_ids.add(resolved.input_id)
                return InputResolved(input_id=resolved.input_id)
            await self._wait_for_update()

    async def checkpoint(self) -> None:
        """Prove that this context still owns its exact lease generation."""
        self._ensure_active()
        await self._store.get_control_snapshot(self._token)

    async def flush_output(self) -> None:
        """Flush buffered output before worker-owned terminal or detach boundaries."""
        self._ensure_active()
        await self._chunker.flush()

    async def detach(self, signal: LeaseLost | RuntimeShutdown) -> None:
        """Flush if still fenced, then close authoritative callbacks and wake controls."""
        if self._detached_signal is not None:
            return
        with suppress(StaleLeaseError):
            await self._chunker.flush()
        self._detached_signal = signal
        self._active = False
        self._detached.set()

    async def wait_for_detach_delivery(self, timeout_seconds: float) -> None:
        """Give a waiting backend one bounded chance to distinguish the detach reason."""
        with suppress(TimeoutError):
            await asyncio.wait_for(self._detach_delivered.wait(), timeout=timeout_seconds)

    async def _append_event(self, event: BackendEvent) -> None:
        self._ensure_active()
        await self._store.append_events(self._token, (event,))
        self._notifier.notify()

    async def _wait_for_update(self) -> None:
        revision = self._notifier.revision
        notifier_task = asyncio.create_task(self._notifier.wait_for_change(revision))
        poll_task = asyncio.create_task(asyncio.sleep(self._control_poll_seconds))
        detach_task = asyncio.create_task(self._detached.wait())
        tasks = (notifier_task, poll_task, detach_task)
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _ensure_active(self) -> None:
        if not self._active:
            raise StaleLeaseError(self._token.job_id, self._token.generation)


def _split_utf8_prefix(text: str, max_bytes: int) -> tuple[str, str]:
    size = 0
    split_at = 0
    for index, character in enumerate(text, start=1):
        character_size = len(character.encode("utf-8"))
        if size + character_size > max_bytes:
            break
        size += character_size
        split_at = index
    if split_at == 0:
        raise ValueError("max_bytes cannot contain one UTF-8 code point")
    return text[:split_at], text[split_at:]


def _sanitize_event(event: BackendEvent) -> BackendEvent:
    allowed = _PAYLOAD_FIELDS.get(event.type, frozenset())
    if (
        event.provider_event_type is not None
        and "reasoning" in event.provider_event_type.casefold()
    ):
        allowed &= _REASONING_METADATA_FIELDS
    payload = {
        key: _normalize_event_scalar(event, key, value)
        for key, value in event.payload.items()
        if key in allowed and _is_scalar(value)
    }
    return BackendEvent(
        type=event.type,
        payload=payload,
        occurred_at=event.occurred_at,
        provider_event_type=event.provider_event_type,
        provider_reference=event.provider_reference,
    )


def _is_scalar(value: JsonValue) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _normalize_event_scalar(event: BackendEvent, key: str, value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        normalized = _normalize_utf8(value)
        if event.type == "message" and key == "text" and event.payload.get("final") is True:
            return normalized
        return normalized[:4096]
    return value


def _normalize_utf8(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return normalized.encode("utf-8", errors="replace").decode("utf-8")
