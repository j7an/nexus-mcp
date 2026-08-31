"""Scripted backend doubles for worker and lifecycle tests."""

import asyncio
import base64
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self

from nexus_mcp.backends import (
    BackendExecutionContext,
    BackendFailure,
    ReconciliationOutcome,
)
from nexus_mcp.core import (
    TERMINAL_STATES,
    AgentJob,
    AgentOperation,
    AgentSession,
    BackendAvailability,
    BackendCapabilities,
    BackendDescriptor,
    BackendEvent,
    CancelReceipt,
    IdempotencyConflictError,
    InputAlreadyResolvedError,
    InputNotFoundError,
    InputRequest,
    InputResolutionReceipt,
    JobAttempt,
    JobError,
    JobEvent,
    JobHandle,
    JobNotFoundError,
    JobResultEnvelope,
    JobState,
    OperationKind,
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
    JobStore,
    LeaseToken,
    PrunePolicy,
    PruneResult,
    ResolveInputCommand,
    RuntimeLease,
    RuntimeLeaseBusyError,
    StoredJobPage,
    SucceededTerminalOutcome,
    TerminalOutcome,
    _create_job_request_hash,
)

__all__ = [
    "EmitEventAction",
    "EmitOutputAction",
    "InMemoryJobStore",
    "RaiseFailureAction",
    "RecordReferenceAction",
    "RequestInputAction",
    "ReturnReconciliationAction",
    "ReturnResultAction",
    "ScriptedBackend",
]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be a timezone-aware UTC datetime")
    return value.astimezone(UTC)


class InMemoryJobStore:
    """Contract-complete single-lock job store for worker and service tests."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._workspaces: dict[str, Workspace] = {}
        self._workspace_ids_by_path: dict[str, str] = {}
        self._sessions: dict[str, AgentSession] = {}
        self._jobs: dict[str, AgentJob] = {}
        self._attempts: dict[str, list[JobAttempt]] = {}
        self._provider_references: dict[str, list[ProviderReference]] = {}
        self._inputs: dict[str, dict[str, PendingInput]] = {}
        self._events: dict[str, list[JobEvent]] = {}
        self._event_sequences: dict[str, int] = {}
        self._results: dict[str, JobResultEnvelope | JobError] = {}
        self._idempotency: dict[tuple[str, str, str, str, str | None], tuple[str, str]] = {}
        self._runtime_leases: dict[str, RuntimeLease] = {}
        self._runtime_generations: dict[str, int] = {}

    async def open(self) -> None:
        """Open the in-memory lifecycle without external resources."""
        async with self._lock:
            pass

    async def close(self) -> None:
        """Close the in-memory lifecycle without discarding persisted test state."""
        async with self._lock:
            pass

    async def resolve_workspace(self, selector: WorkspaceSelector) -> Workspace:
        """Resolve a workspace by durable id or exact canonical path."""
        async with self._lock:
            if selector.workspace_id is not None:
                workspace = self._workspaces.get(selector.workspace_id)
                identity = selector.workspace_id
            else:
                assert selector.path is not None
                identity = str(selector.path)
                workspace_id = self._workspace_ids_by_path.get(identity)
                workspace = None if workspace_id is None else self._workspaces.get(workspace_id)
            if workspace is None:
                raise WorkspaceInvalidError(identity, "workspace is not registered")
            return workspace

    async def create_job(self, command: CreateJobCommand) -> CreateJobResult:
        """Atomically persist one admitted job, session if needed, and queued event."""
        async with self._lock:
            self._validate_source_checkpoint(command)
            request_hash = _create_job_request_hash(command)
            idempotency_scope = self._idempotency_scope(command)
            if idempotency_scope is not None:
                existing = self._idempotency.get(idempotency_scope)
                if existing is not None:
                    existing_hash, existing_job_id = existing
                    if existing_hash != request_hash:
                        assert command.idempotency_key is not None
                        raise IdempotencyConflictError(command.idempotency_key)
                    job = self._jobs[existing_job_id]
                    return CreateJobResult(handle=self._handle(job), created=False)

            active = self._active_session_job(command.session_id)
            if active is not None:
                assert command.session_id is not None
                raise SessionBusyError(command.session_id, active.job_id)

            new_session = self._prepare_session(command)
            self._persist_workspace(command.workspace)
            if new_session is not None:
                self._sessions[new_session.session_id] = new_session
            now = _utc_now()
            job = AgentJob(
                job_id=new_id(),
                workspace_id=command.workspace.workspace_id,
                backend_id=command.backend_id,
                owner_id=command.owner_id,
                access_policy=command.access_policy,
                operation=command.operation,
                requested_config=command.requested_config,
                request_hash=request_hash,
                session_id=command.session_id,
                idempotency_key=command.idempotency_key,
                source_checkpoint=command.source_checkpoint,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job.job_id] = job
            self._attempts[job.job_id] = []
            self._provider_references[job.job_id] = list(command.source_checkpoint)
            self._inputs[job.job_id] = {}
            self._events[job.job_id] = []
            self._event_sequences[job.job_id] = 0
            self._commit_event(job.job_id, command.queued_event)
            if idempotency_scope is not None:
                self._idempotency[idempotency_scope] = (request_hash, job.job_id)
            return CreateJobResult(handle=self._handle(job), created=True)

    async def get_session(self, session_id: str) -> AgentSession | None:
        """Return one durable session snapshot when present."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_job(self, job_id: str) -> AgentJob | None:
        """Return one durable job snapshot when present."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def get_job_result(self, job_id: str) -> JobResultEnvelope | JobError | None:
        """Return the normalized successful result or terminal error when present."""
        async with self._lock:
            return self._results.get(job_id)

    async def get_job_attempts(self, job_id: str) -> tuple[JobAttempt, ...]:
        """Return persisted attempts in ascending attempt order."""
        async with self._lock:
            return tuple(self._attempts.get(job_id, ()))

    async def get_provider_references(
        self, *, session_id: str | None = None, job_id: str | None = None
    ) -> tuple[ProviderReference, ...]:
        """Return references scoped to exactly one job or session."""
        async with self._lock:
            if (session_id is None) == (job_id is None):
                raise ValueError("exactly one of session_id or job_id is required")
            if job_id is not None:
                return tuple(self._provider_references.get(job_id, ()))
            assert session_id is not None
            session = self._sessions.get(session_id)
            return () if session is None else session.provider_references

    async def get_pending_inputs(self, job_id: str) -> tuple[PendingInput, ...]:
        """Return only unresolved inputs in stable creation order."""
        async with self._lock:
            records = self._inputs.get(job_id, {})
            return tuple(item for item in records.values() if item.response is None)

    async def list_jobs(self, query: JobQuery) -> StoredJobPage:
        """Return one authorized descending job page with an opaque pair cursor."""
        async with self._lock:
            jobs = [
                job
                for job in self._jobs.values()
                if job.workspace_id == query.workspace_id
                and job.state in query.states
                and self._job_is_visible(job, query)
            ]
            jobs.sort(key=lambda job: (job.created_at, job.job_id), reverse=True)
            if query.cursor is not None:
                cursor = self._decode_cursor(query.cursor)
                jobs = [job for job in jobs if (job.created_at, job.job_id) < cursor]
            page = jobs[: query.limit]
            next_cursor = None
            if len(jobs) > query.limit:
                final = page[-1]
                next_cursor = self._encode_cursor(final.created_at, final.job_id)
            return StoredJobPage(jobs=tuple(page), next_cursor=next_cursor)

    async def claim_next(
        self,
        owner_id: str,
        lease_until: datetime,
        *,
        event: BackendEvent,
    ) -> ClaimedJob | None:
        """Claim the oldest eligible job and allocate a new fencing generation."""
        lease_until = _normalize_utc(lease_until)
        async with self._lock:
            now = _utc_now()
            eligible = [job for job in self._jobs.values() if self._claimable(job, now)]
            if not eligible:
                return None
            job = min(eligible, key=lambda item: (item.created_at, item.job_id))
            previous_attempts = self._attempts[job.job_id]
            safe_retry = bool(
                previous_attempts and previous_attempts[-1].retry_classification == "safe_to_retry"
            )
            if previous_attempts and previous_attempts[-1].ended_at is None:
                previous_attempts[-1] = previous_attempts[-1].model_copy(update={"ended_at": now})
            generation = (job.lease_generation or 0) + 1
            attempt_number = len(previous_attempts) + 1
            reclaimed = job.lease_generation is not None
            attempt = JobAttempt(
                job_id=job.job_id,
                attempt_number=attempt_number,
                phase="executing" if safe_retry else ("reconciling" if reclaimed else "claiming"),
                worker_id=owner_id,
                lease_generation=generation,
                lease_expires_at=lease_until,
                heartbeat_at=now,
                started_at=now,
            )
            previous_attempts.append(attempt)
            job = job.model_copy(
                update={
                    "lease_owner_id": owner_id,
                    "lease_generation": generation,
                    "lease_expires_at": lease_until,
                    "retry_at": None,
                    "updated_at": now,
                }
            )
            self._jobs[job.job_id] = job
            token = LeaseToken(
                job_id=job.job_id,
                owner_id=owner_id,
                generation=generation,
                attempt_number=attempt_number,
            )
            self._commit_event(job.job_id, event, attempt_number=attempt_number)
            return ClaimedJob(job=job, attempt=attempt, token=token)

    async def renew_lease(self, token: LeaseToken, lease_until: datetime) -> bool:
        """Extend only the current unexpired job fence."""
        lease_until = _normalize_utc(lease_until)
        async with self._lock:
            if not self._token_matches(token):
                return False
            now = _utc_now()
            job = self._jobs[token.job_id].model_copy(
                update={"lease_expires_at": lease_until, "updated_at": now}
            )
            self._jobs[token.job_id] = job
            self._replace_attempt(
                token,
                self._attempt_for(token).model_copy(
                    update={"lease_expires_at": lease_until, "heartbeat_at": now}
                ),
            )
            return True

    async def store_resolved_config(
        self, token: LeaseToken, config: ResolvedExecutionConfig
    ) -> None:
        """Freeze effective configuration under the current worker fence."""
        async with self._lock:
            job = self._require_token(token)
            if job.resolved_config is not None and job.resolved_config != config:
                raise ValueError("resolved execution config is already stored")
            self._jobs[job.job_id] = job.model_copy(
                update={"resolved_config": config, "updated_at": _utc_now()}
            )
            if (trace := getattr(self, "trace", None)) is not None:
                trace.append("store_resolved_config")

    async def record_provider_reference(
        self, token: LeaseToken, reference: ProviderReference
    ) -> None:
        """Persist one provider identifier at job and session scope exactly once."""
        async with self._lock:
            job = self._require_token(token)
            references = self._provider_references[job.job_id]
            if reference not in references:
                references.append(reference)
            if job.session_id is not None:
                session = self._sessions[job.session_id]
                if reference not in session.provider_references:
                    self._sessions[job.session_id] = session.model_copy(
                        update={
                            "provider_references": (*session.provider_references, reference),
                            "updated_at": _utc_now(),
                        }
                    )

    async def append_events(
        self, token: LeaseToken, events: tuple[BackendEvent, ...]
    ) -> tuple[JobEvent, ...]:
        """Allocate consecutive event sequences under the current worker fence."""
        async with self._lock:
            self._require_token(token)
            return tuple(
                self._commit_event(token.job_id, event, attempt_number=token.attempt_number)
                for event in events
            )

    async def mark_input_required(
        self,
        token: LeaseToken,
        inputs: tuple[PendingInput, ...],
        *,
        event: BackendEvent,
    ) -> None:
        """Atomically persist new unresolved inputs, public state, and one event."""
        if not inputs:
            raise ValueError("at least one pending input is required")
        async with self._lock:
            job = self._require_token(token)
            validate_job_transition(job.state, "input_required")
            input_ids = [item.input_id for item in inputs]
            if len(input_ids) != len(set(input_ids)):
                raise ValueError("pending input ids must be unique")
            records = self._inputs[job.job_id]
            for item in inputs:
                if item.job_id != job.job_id:
                    raise ValueError("pending input belongs to another job")
                if item.input_id in records:
                    raise ValueError(f"pending input already exists: {item.input_id}")
            records.update({item.input_id: item for item in inputs})
            now = _utc_now()
            self._jobs[job.job_id] = job.model_copy(
                update={"state": "input_required", "updated_at": now}
            )
            attempt = self._attempt_for(token).model_copy(update={"phase": "executing"})
            self._replace_attempt(token, attempt)
            self._commit_event(job.job_id, event, attempt_number=token.attempt_number)

    async def mark_running(
        self,
        token: LeaseToken,
        resolved_input_ids: tuple[str, ...],
        *,
        event: BackendEvent,
    ) -> None:
        """Atomically enter or resume running after every named input is resolved."""
        async with self._lock:
            job = self._require_token(token)
            validate_job_transition(job.state, "running")
            records = self._inputs[job.job_id]
            if len(resolved_input_ids) != len(set(resolved_input_ids)):
                raise ValueError("resolved input ids must be unique")
            for input_id in resolved_input_ids:
                item = records.get(input_id)
                if item is None or item.response is None:
                    raise InputNotFoundError(job.job_id, input_id)
            if job.state == "input_required" and any(
                item.response is None for item in records.values()
            ):
                raise ValueError("all pending inputs must be resolved before resuming")
            now = _utc_now()
            self._jobs[job.job_id] = job.model_copy(
                update={"state": "running", "retry_at": None, "updated_at": now}
            )
            self._replace_attempt(
                token,
                self._attempt_for(token).model_copy(update={"phase": "executing"}),
            )
            self._commit_event(job.job_id, event, attempt_number=token.attempt_number)

    async def mark_reconciling(
        self,
        token: LeaseToken,
        error: JobError,
        *,
        event: BackendEvent,
    ) -> None:
        """Record uncertain provider state and its semantic event under one fence."""
        async with self._lock:
            job = self._require_token(token)
            attempt = self._attempt_for(token).model_copy(
                update={
                    "phase": "reconciling",
                    "reconciliation_classification": error.code,
                    "error_code": error.code,
                    "error_message": error.message,
                }
            )
            self._replace_attempt(token, attempt)
            self._jobs[job.job_id] = job.model_copy(update={"updated_at": _utc_now()})
            self._commit_event(job.job_id, event, attempt_number=token.attempt_number)

    async def schedule_retry(
        self,
        token: LeaseToken,
        retry_at: datetime,
        error: JobError,
        *,
        event: BackendEvent,
    ) -> None:
        """Persist retry backoff, close the attempt, and relinquish its lease atomically."""
        retry_at = _normalize_utc(retry_at)
        async with self._lock:
            job = self._require_token(token)
            now = _utc_now()
            attempt = self._attempt_for(token).model_copy(
                update={
                    "phase": "finalizing",
                    "retry_classification": error.retry_disposition,
                    "error_code": error.code,
                    "error_message": error.message,
                    "ended_at": now,
                }
            )
            self._replace_attempt(token, attempt)
            self._commit_event(job.job_id, event, attempt_number=token.attempt_number)
            self._jobs[job.job_id] = job.model_copy(
                update={
                    "lease_owner_id": None,
                    "lease_expires_at": None,
                    "retry_at": retry_at,
                    "updated_at": now,
                }
            )

    async def get_control_snapshot(self, token: LeaseToken) -> ControlSnapshot:
        """Return current control facts only to the matching lease generation."""
        async with self._lock:
            job = self._require_token(token)
            inputs = tuple(self._inputs[job.job_id].values())
            return ControlSnapshot(
                state=job.state,
                cancel_requested=job.cancel_requested_at is not None,
                unresolved_inputs=tuple(item for item in inputs if item.response is None),
                resolved_inputs=tuple(item for item in inputs if item.response is not None),
                lease_generation=token.generation,
            )

    async def resolve_input(self, command: ResolveInputCommand) -> InputResolutionReceipt:
        """Persist one validated response idempotently with exactly one semantic event."""
        async with self._lock:
            job = self._jobs.get(command.job_id)
            if job is None:
                raise JobNotFoundError(command.job_id)
            pending = self._inputs[job.job_id].get(command.input_id)
            if pending is None:
                raise InputNotFoundError(job.job_id, command.input_id)
            response = pending.validate_response(command.response)
            if pending.response is not None:
                if pending.response != response:
                    raise InputAlreadyResolvedError(job.job_id, pending.input_id)
                return InputResolutionReceipt(
                    job_id=job.job_id,
                    input_id=pending.input_id,
                    replayed=True,
                )
            if job.state != "input_required":
                raise InputNotFoundError(job.job_id, command.input_id)
            resolved = PendingInput(
                input_id=pending.input_id,
                job_id=pending.job_id,
                request=pending.request,
                provider_reference=pending.provider_reference,
                created_at=pending.created_at,
                resolved_at=command.resolved_at,
                response=response,
            )
            self._inputs[job.job_id][pending.input_id] = resolved
            attempt_number = self._current_attempt_number(job.job_id)
            self._commit_event(job.job_id, command.event, attempt_number=attempt_number)
            return InputResolutionReceipt(job_id=job.job_id, input_id=pending.input_id)

    async def request_cancel(self, command: CancelJobCommand) -> CancelReceipt:
        """Persist cancellation intent once; queued jobs become terminal immediately."""
        async with self._lock:
            job = self._jobs.get(command.job_id)
            if job is None:
                raise JobNotFoundError(command.job_id)
            if job.state in TERMINAL_STATES:
                return CancelReceipt(
                    job_id=job.job_id,
                    state=job.state,
                    cancel_requested=job.cancel_requested_at is not None,
                    completed_immediately=job.state == "cancelled",
                    event_committed=False,
                )
            if job.cancel_requested_at is not None:
                return CancelReceipt(
                    job_id=job.job_id,
                    state=job.state,
                    cancel_requested=True,
                    completed_immediately=False,
                    event_committed=False,
                )
            if job.state != "queued" and not command.active_cancellation_allowed:
                return CancelReceipt(
                    job_id=job.job_id,
                    state=job.state,
                    cancel_requested=False,
                    completed_immediately=False,
                    event_committed=False,
                )
            now = command.requested_at
            updates: dict[str, object] = {"cancel_requested_at": now, "updated_at": now}
            completed_immediately = job.state == "queued"
            if completed_immediately:
                validate_job_transition(job.state, "cancelled")
                updates |= {
                    "state": "cancelled",
                    "completed_at": now,
                    "lease_owner_id": None,
                    "lease_expires_at": None,
                }
                attempts = self._attempts[job.job_id]
                if attempts and attempts[-1].ended_at is None:
                    attempts[-1] = attempts[-1].model_copy(
                        update={"phase": "finalizing", "ended_at": now}
                    )
            job = job.model_copy(update=updates)
            self._jobs[job.job_id] = job
            self._commit_event(
                job.job_id,
                command.queued_event if completed_immediately else command.active_event,
                attempt_number=self._current_attempt_number(job.job_id),
            )
            return CancelReceipt(
                job_id=job.job_id,
                state=job.state,
                cancel_requested=True,
                completed_immediately=completed_immediately,
                event_committed=True,
            )

    async def terminalize(
        self,
        token: LeaseToken,
        outcome: TerminalOutcome,
        *,
        event: BackendEvent,
    ) -> AgentJob:
        """Atomically write one terminal snapshot, normalized result, and final event."""
        async with self._lock:
            job = self._require_token(token)
            target: JobState
            match outcome:
                case SucceededTerminalOutcome(result=result, completed_at=completed_at):
                    if result.kind != job.operation.kind:
                        raise ValueError("terminal result kind does not match job operation")
                    target = "completed"
                    stored_result: JobResultEnvelope | JobError | None = JobResultEnvelope(
                        job_id=job.job_id,
                        payload=result,
                        completed_at=completed_at,
                    )
                case FailedTerminalOutcome(error=error, completed_at=completed_at):
                    target = "failed"
                    stored_result = error
                case CancelledTerminalOutcome(completed_at=completed_at):
                    target = "cancelled"
                    stored_result = None
            validate_job_transition(job.state, target)
            now = completed_at
            job = job.model_copy(
                update={
                    "state": target,
                    "completed_at": completed_at,
                    "updated_at": now,
                    "lease_owner_id": None,
                    "lease_expires_at": None,
                    "retry_at": None,
                }
            )
            self._jobs[job.job_id] = job
            if stored_result is not None:
                self._results[job.job_id] = stored_result
            attempt_updates: dict[str, object] = {
                "phase": "finalizing",
                "ended_at": completed_at,
            }
            if isinstance(outcome, FailedTerminalOutcome):
                attempt_updates |= {
                    "error_code": outcome.error.code,
                    "error_message": outcome.error.message,
                }
            attempt = self._attempt_for(token).model_copy(update=attempt_updates)
            self._replace_attempt(token, attempt)
            self._commit_event(job.job_id, event, attempt_number=token.attempt_number)
            return job

    async def read_events(self, job_id: str, after_sequence: int, limit: int) -> EventPage:
        """Return committed events strictly after a bounded sequence cursor."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if not 1 <= limit <= 1000:
            raise ValueError("event page limit must be from 1 through 1000")
        async with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            candidates = [
                event for event in self._events[job_id] if event.sequence > after_sequence
            ]
            page = tuple(candidates[:limit])
            return EventPage(
                events=page,
                next_after_sequence=None if not page else page[-1].sequence,
                has_more=len(candidates) > limit,
            )

    async def acquire_runtime_lease(
        self, runtime_key: str, owner_id: str, lease_until: datetime
    ) -> RuntimeLease:
        """Acquire, idempotently extend, or generation-fence one runtime owner."""
        lease_until = _normalize_utc(lease_until)
        async with self._lock:
            now = _utc_now()
            current = self._runtime_leases.get(runtime_key)
            if current is not None and current.lease_until > now:
                if current.owner_id != owner_id:
                    raise RuntimeLeaseBusyError(
                        runtime_key,
                        current.owner_id,
                        current.lease_until,
                    )
                lease = current.model_copy(
                    update={
                        "lease_until": max(current.lease_until, lease_until),
                        "heartbeat_at": now,
                    }
                )
            else:
                generation = self._runtime_generations.get(runtime_key, 0) + 1
                lease = RuntimeLease(
                    runtime_key=runtime_key,
                    owner_id=owner_id,
                    generation=generation,
                    lease_until=lease_until,
                    heartbeat_at=now,
                )
            self._runtime_leases[runtime_key] = lease
            self._runtime_generations[runtime_key] = lease.generation
            return lease

    async def renew_runtime_lease(self, lease: RuntimeLease, lease_until: datetime) -> bool:
        """Renew and persist endpoint data only for the matching runtime token."""
        lease_until = _normalize_utc(lease_until)
        async with self._lock:
            current = self._runtime_leases.get(lease.runtime_key)
            if (
                current is None
                or current.lease_until <= _utc_now()
                or (current.owner_id != lease.owner_id or current.generation != lease.generation)
            ):
                return False
            self._runtime_leases[lease.runtime_key] = current.model_copy(
                update={
                    "lease_until": lease_until,
                    "heartbeat_at": _utc_now(),
                    "endpoint": lease.endpoint,
                }
            )
            return True

    async def release_runtime_lease(self, lease: RuntimeLease) -> None:
        """Release only the matching runtime generation; stale releases are no-ops."""
        async with self._lock:
            current = self._runtime_leases.get(lease.runtime_key)
            if (
                current is not None
                and current.lease_until > _utc_now()
                and (current.owner_id == lease.owner_id and current.generation == lease.generation)
            ):
                del self._runtime_leases[lease.runtime_key]

    async def prune(self, policy: PrunePolicy, now: datetime) -> PruneResult:
        """Delete eligible terminal snapshots and old events without touching active jobs."""
        now = _normalize_utc(now)
        async with self._lock:
            terminal_ids: set[str] = set()
            if policy.terminal_job_before is not None:
                cutoff = min(policy.terminal_job_before, now)
                terminal_ids = {
                    job.job_id
                    for job in self._jobs.values()
                    if job.state in TERMINAL_STATES
                    and job.completed_at is not None
                    and job.completed_at < cutoff
                    and not any(item.response is None for item in self._inputs[job.job_id].values())
                }
            events_deleted = sum(len(self._events.get(job_id, ())) for job_id in terminal_ids)
            for job_id in terminal_ids:
                self._delete_job(job_id)

            if policy.event_before is not None:
                cutoff = min(policy.event_before, now)
                for job_id, events in self._events.items():
                    retained = [event for event in events if event.occurred_at >= cutoff]
                    events_deleted += len(events) - len(retained)
                    self._events[job_id] = retained
            return PruneResult(
                terminal_jobs_deleted=len(terminal_ids),
                events_deleted=events_deleted,
                raw_diagnostics_deleted=0,
            )

    def _persist_workspace(self, workspace: Workspace) -> None:
        existing = self._workspaces.get(workspace.workspace_id)
        if existing is not None and existing.canonical_path != workspace.canonical_path:
            raise WorkspaceInvalidError(workspace.workspace_id, "workspace path changed")
        path = str(workspace.canonical_path)
        existing_id = self._workspace_ids_by_path.get(path)
        if existing_id is not None and existing_id != workspace.workspace_id:
            raise WorkspaceInvalidError(path, "canonical path belongs to another workspace")
        self._workspaces[workspace.workspace_id] = workspace
        self._workspace_ids_by_path[path] = workspace.workspace_id

    def _validate_source_checkpoint(self, command: CreateJobCommand) -> None:
        if not command.source_checkpoint:
            return
        source_session_id = command.source_session_id
        assert source_session_id is not None
        source_session = self._sessions.get(source_session_id)
        if source_session is None:
            raise SessionNotFoundError(source_session_id)
        if any(
            reference not in source_session.provider_references
            for reference in command.source_checkpoint
        ):
            raise ValueError("source checkpoint contains a reference not owned by its session")

    def _prepare_session(self, command: CreateJobCommand) -> AgentSession | None:
        if command.session_id is None:
            return None
        existing = self._sessions.get(command.session_id)
        if existing is None:
            if not command.create_session:
                raise SessionNotFoundError(command.session_id)
            if (
                command.parent_session_id is not None
                and command.parent_session_id not in self._sessions
            ):
                raise SessionNotFoundError(command.parent_session_id)
            return AgentSession(
                session_id=command.session_id,
                workspace_id=command.workspace.workspace_id,
                backend_id=command.backend_id,
                owner_id=command.owner_id,
                access_policy=command.access_policy,
                parent_session_id=command.parent_session_id,
            )

        expected = (
            command.workspace.workspace_id,
            command.backend_id,
            command.owner_id,
            command.access_policy,
            command.parent_session_id,
        )
        actual = (
            existing.workspace_id,
            existing.backend_id,
            existing.owner_id,
            existing.access_policy,
            existing.parent_session_id,
        )
        if actual != expected:
            raise ValueError("session identity does not match the admitted request")
        return None

    def _active_session_job(self, session_id: str | None) -> AgentJob | None:
        if session_id is None:
            return None
        return next(
            (
                job
                for job in self._jobs.values()
                if job.session_id == session_id and job.state not in TERMINAL_STATES
            ),
            None,
        )

    @staticmethod
    def _idempotency_scope(
        command: CreateJobCommand,
    ) -> tuple[str, str, str, str, str | None] | None:
        if command.idempotency_key is None:
            return None
        return (
            command.owner_id,
            command.workspace.workspace_id,
            command.command_family,
            command.idempotency_key,
            command.source_session_id,
        )

    @staticmethod
    def _handle(job: AgentJob) -> JobHandle:
        return JobHandle(
            job_id=job.job_id,
            session_id=job.session_id,
            operation=job.operation,
        )

    def _commit_event(
        self,
        job_id: str,
        event: BackendEvent,
        *,
        attempt_number: int | None = None,
    ) -> JobEvent:
        events = self._events[job_id]
        sequence = self._event_sequences[job_id] + 1
        committed = JobEvent(
            job_id=job_id,
            sequence=sequence,
            type=event.type,
            payload=event.payload,
            occurred_at=event.occurred_at,
            attempt_number=attempt_number,
            provider_event_type=event.provider_event_type,
            provider_reference=event.provider_reference,
        )
        events.append(committed)
        self._event_sequences[job_id] = sequence
        return committed

    def _claimable(self, job: AgentJob, now: datetime) -> bool:
        if job.state in TERMINAL_STATES:
            return False
        lease_available = job.lease_expires_at is None or job.lease_expires_at <= now
        retry_due = job.retry_at is None or job.retry_at <= now
        if job.state == "queued":
            return lease_available and retry_due
        return lease_available and retry_due

    def _token_matches(self, token: LeaseToken) -> bool:
        job = self._jobs.get(token.job_id)
        if job is None or job.state in TERMINAL_STATES:
            return False
        return (
            job.lease_owner_id == token.owner_id
            and job.lease_generation == token.generation
            and job.lease_expires_at is not None
            and job.lease_expires_at > _utc_now()
            and bool(self._attempts.get(token.job_id))
            and self._attempts[token.job_id][-1].attempt_number == token.attempt_number
        )

    def _require_token(self, token: LeaseToken) -> AgentJob:
        if not self._token_matches(token):
            raise StaleLeaseError(token.job_id, token.generation)
        return self._jobs[token.job_id]

    def _attempt_for(self, token: LeaseToken) -> JobAttempt:
        for attempt in self._attempts[token.job_id]:
            if attempt.attempt_number == token.attempt_number:
                return attempt
        raise StaleLeaseError(token.job_id, token.generation)

    def _replace_attempt(self, token: LeaseToken, replacement: JobAttempt) -> None:
        attempts = self._attempts[token.job_id]
        for index, attempt in enumerate(attempts):
            if attempt.attempt_number == token.attempt_number:
                attempts[index] = replacement
                return
        raise StaleLeaseError(token.job_id, token.generation)

    def _current_attempt_number(self, job_id: str) -> int | None:
        attempts = self._attempts.get(job_id, ())
        return None if not attempts else attempts[-1].attempt_number

    @staticmethod
    def _job_is_visible(job: AgentJob, query: JobQuery) -> bool:
        return job.owner_id == query.access.principal_id or (
            job.access_policy == "workspace" and query.access.workspace_authorized
        )

    @staticmethod
    def _encode_cursor(created_at: datetime, job_id: str) -> str:
        raw = json.dumps([created_at.isoformat(), job_id], separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str) -> tuple[datetime, str]:
        try:
            padding = "=" * (-len(cursor) % 4)
            raw = base64.urlsafe_b64decode(cursor + padding)
            created_at_raw, job_id = json.loads(raw)
            created_at = _normalize_utc(datetime.fromisoformat(created_at_raw))
            if not isinstance(job_id, str) or not job_id:
                raise ValueError
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("invalid stored-job cursor") from error
        return created_at, job_id

    def _delete_job(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._attempts.pop(job_id, None)
        self._provider_references.pop(job_id, None)
        self._inputs.pop(job_id, None)
        self._events.pop(job_id, None)
        self._event_sequences.pop(job_id, None)
        self._results.pop(job_id, None)
        self._idempotency = {
            scope: value for scope, value in self._idempotency.items() if value[1] != job_id
        }


if TYPE_CHECKING:
    _JOB_STORE_CONTRACT: JobStore = InMemoryJobStore()


@dataclass(frozen=True)
class EmitEventAction:
    """Tell a scripted backend to emit one normalized event."""

    event: BackendEvent


@dataclass(frozen=True)
class EmitOutputAction:
    """Tell a scripted backend to emit one incremental message delta."""

    text: str


@dataclass(frozen=True)
class RecordReferenceAction:
    """Tell a scripted backend to record one provider reference."""

    reference: ProviderReference


@dataclass(frozen=True)
class RequestInputAction:
    """Tell a scripted backend to request one normalized input."""

    request: InputRequest


@dataclass(frozen=True)
class RaiseFailureAction:
    """Tell a scripted backend to raise one classified backend failure."""

    failure: BackendFailure


@dataclass(frozen=True)
class ReturnResultAction:
    """Tell a scripted backend to return one normalized operation result."""

    result: OperationResult


@dataclass(frozen=True)
class ReturnReconciliationAction:
    """Tell a scripted backend to return one reconciliation decision."""

    outcome: ReconciliationOutcome


type ContextAction = EmitEventAction | EmitOutputAction | RecordReferenceAction | RequestInputAction
type ExecutionAction = ContextAction | RaiseFailureAction | ReturnResultAction
type ReconciliationAction = ContextAction | RaiseFailureAction | ReturnReconciliationAction


class ScriptedBackend:
    """A deterministic backend whose externally visible effects are supplied by test scripts."""

    def __init__(self, backend_id: str = "scripted") -> None:
        self.descriptor = BackendDescriptor(
            backend_id=backend_id,
            display_name="Scripted Backend",
            capabilities=BackendCapabilities(
                operations=frozenset({"turn", "fork", "review", "diagnostics"}),
                cancellation=True,
                graceful_interrupt=True,
                session_fork=True,
                input_required=True,
            ),
        )
        self.availability = BackendAvailability(available=True)
        self.execute_calls: list[tuple[AgentOperation, BackendExecutionContext]] = []
        self.reconcile_calls: list[
            tuple[tuple[ProviderReference, ...], BackendExecutionContext]
        ] = []
        self.config_calls: list[tuple[RequestedExecutionConfig, Workspace]] = []
        self.close_calls = 0
        self.trace: list[str] | None = None
        self.input_requests: list[InputRequest] = []
        self.input_responses: list[object] = []
        self._execution_actions: deque[ExecutionAction] = deque()
        self._reconciliation_actions: deque[ReconciliationAction] = deque()

    @property
    def backend_id(self) -> str:
        """Return the stable registered identifier."""
        return self.descriptor.backend_id

    def with_operations(self, operations: Iterable[OperationKind]) -> Self:
        """Replace advertised operations while preserving every other static capability."""
        self.descriptor = self.descriptor.model_copy(
            update={
                "capabilities": self.descriptor.capabilities.model_copy(
                    update={"operations": frozenset(operations)}
                )
            }
        )
        return self

    def queue_execute(self, *actions: ExecutionAction) -> None:
        """Append exact effects and one eventual result or failure for ``execute``."""
        self._execution_actions.extend(actions)

    def queue_reconcile(self, *actions: ReconciliationAction) -> None:
        """Append exact effects and one eventual outcome or failure for ``reconcile``."""
        self._reconciliation_actions.extend(actions)

    async def check_availability(self, workspace: Workspace) -> BackendAvailability:
        """Return the configured health observation without mutating registration state."""
        self._trace("backend.check_availability")
        return self.availability

    async def resolve_execution_config(
        self, requested: RequestedExecutionConfig, workspace: Workspace
    ) -> ResolvedExecutionConfig:
        """Record and resolve the requested configuration with no provider defaults."""
        self._trace("backend.resolve_execution_config")
        self.config_calls.append((requested, workspace))
        return ResolvedExecutionConfig.from_requested(requested, backend_defaults={})

    async def execute(
        self, operation: AgentOperation, context: BackendExecutionContext
    ) -> OperationResult:
        """Apply scripted effects until one exact result or classified failure is reached."""
        self._trace("backend.execute")
        self.execute_calls.append((operation, context))
        while self._execution_actions:
            action = self._execution_actions.popleft()
            match action:
                case (
                    EmitEventAction()
                    | EmitOutputAction()
                    | RecordReferenceAction()
                    | RequestInputAction()
                ):
                    await self._apply_context_action(action, context)
                case RaiseFailureAction(failure=failure):
                    raise failure
                case ReturnResultAction(result=result):
                    return result
        raise AssertionError("Scripted backend execute() has no queued outcome")

    async def reconcile(
        self,
        provider_state: tuple[ProviderReference, ...],
        context: BackendExecutionContext,
    ) -> ReconciliationOutcome:
        """Apply scripted effects until one exact reconciliation outcome or failure is reached."""
        self._trace("backend.reconcile")
        self.reconcile_calls.append((provider_state, context))
        while self._reconciliation_actions:
            action = self._reconciliation_actions.popleft()
            match action:
                case (
                    EmitEventAction()
                    | EmitOutputAction()
                    | RecordReferenceAction()
                    | RequestInputAction()
                ):
                    await self._apply_context_action(action, context)
                case RaiseFailureAction(failure=failure):
                    raise failure
                case ReturnReconciliationAction(outcome=outcome):
                    return outcome
        raise AssertionError("Scripted backend reconcile() has no queued outcome")

    async def close(self) -> None:
        """Record one runtime close request."""
        self.close_calls += 1

    async def _apply_context_action(
        self, action: ContextAction, context: BackendExecutionContext
    ) -> None:
        """Apply one shared context effect without accessing worker-owned state directly."""
        match action:
            case EmitEventAction(event=event):
                await context.emit(event)
            case EmitOutputAction(text=text):
                await context.emit_output_delta(text)
            case RecordReferenceAction(reference=reference):
                await context.record_provider_reference(reference)
            case RequestInputAction(request=request):
                self.input_requests.append(request)
                self.input_responses.append(await context.request_input(request))

    def _trace(self, event: str) -> None:
        if self.trace is not None:
            self.trace.append(event)
