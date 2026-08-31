"""Durable generation-fenced job workers with safe retry and reconciliation."""

import asyncio
import random
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from nexus_mcp.backends import (
    ActiveReconciliationOutcome,
    AgentBackend,
    BackendFailure,
    CancelledReconciliationOutcome,
    CompletedReconciliationOutcome,
    FailedReconciliationOutcome,
    InputRequiredReconciliationOutcome,
    LeaseLost,
    RuntimeShutdown,
    UnknownReconciliationOutcome,
)
from nexus_mcp.backends.manager import BackendManager
from nexus_mcp.core import (
    BackendEvent,
    JobError,
    JobErrorCode,
    NexusCoreError,
    OperationResult,
    ProviderReference,
    RetryPolicy,
    StaleLeaseError,
    TurnResult,
    WorkspaceSelector,
)
from nexus_mcp.jobs.control import StoreBackedExecutionContext
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.store import (
    CancelledTerminalOutcome,
    ClaimedJob,
    FailedTerminalOutcome,
    JobStore,
    LeaseToken,
    SucceededTerminalOutcome,
)

__all__ = [
    "ExponentialRetryDelay",
    "JobWorker",
    "RetryDelay",
    "WorkerPolicy",
    "WorkerPool",
]

type Clock = Callable[[], datetime]

_JOB_ERROR_CODE_ADAPTER: TypeAdapter[JobErrorCode] = TypeAdapter(JobErrorCode)


class WorkerPolicy(BaseModel):
    """Runtime-owned worker lease, polling, and reconciliation timing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lease_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    heartbeat_seconds: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    idle_poll_seconds: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    reconciliation_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def require_safe_heartbeat_window(self) -> "WorkerPolicy":
        """Require more than two heartbeat opportunities inside one lease."""
        if self.heartbeat_seconds * 2 >= self.lease_seconds:
            raise ValueError("heartbeat_seconds * 2 must be less than lease_seconds")
        return self


class RetryDelay(Protocol):
    """Calculate one persisted safe-retry delay without owning any sleeping."""

    def __call__(
        self,
        attempt: int,
        retry_after: float | None,
        policy: RetryPolicy,
    ) -> float: ...


class ExponentialRetryDelay:
    """Calculate bounded full jitter while respecting a provider retry minimum."""

    def __init__(self, random_value: Callable[[], float] = random.random) -> None:
        self._random_value = random_value

    def __call__(
        self,
        attempt: int,
        retry_after: float | None,
        policy: RetryPolicy,
    ) -> float:
        cap = min(
            policy.max_delay_seconds,
            policy.base_delay_seconds * (2 ** max(0, attempt - 1)),
        )
        jitter = max(0.0, min(1.0, self._random_value())) * cap
        return float(max(jitter, retry_after or 0.0))


class JobWorker:
    """Claim and execute one durable job at a time under an exact lease fence."""

    def __init__(
        self,
        *,
        worker_id: str,
        store: JobStore,
        backends: BackendManager,
        notifier: EventNotifier,
        policy: WorkerPolicy | None = None,
        retry_delay: RetryDelay | None = None,
        clock: Clock | None = None,
        output_chunk_bytes: int = 4096,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        self.worker_id = worker_id
        self._store = store
        self._backends = backends
        self._notifier = notifier
        self._policy = policy or WorkerPolicy()
        self._retry_delay = retry_delay or ExponentialRetryDelay()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._output_chunk_bytes = output_chunk_bytes
        self._active_run: asyncio.Task[Any] | None = None
        self._active_context: StoreBackedExecutionContext | None = None
        self._active_observation: asyncio.Task[None] | None = None

    async def run_once(self) -> bool:
        """Claim and fully observe at most one eligible job."""
        claimed = await self._store.claim_next(
            self.worker_id,
            self._now() + timedelta(seconds=self._policy.lease_seconds),
            event=BackendEvent(type="progress", payload={"stage": "claimed"}),
        )
        if claimed is None:
            return False

        self._active_run = asyncio.current_task()
        heartbeat_stop = asyncio.Event()
        lease_lost = asyncio.Event()
        context_holder: list[StoreBackedExecutionContext] = []
        observation_holder: list[asyncio.Task[None]] = []
        heartbeat = asyncio.create_task(
            self._heartbeat(
                claimed.token,
                heartbeat_stop,
                lease_lost,
                context_holder,
                observation_holder,
            )
        )
        try:
            await self._prepare_and_observe(
                claimed,
                lease_lost,
                context_holder,
                observation_holder,
            )
        finally:
            heartbeat_stop.set()
            await heartbeat
            self._active_run = None
            self._active_context = None
            self._active_observation = None
        return True

    async def run(self, stop: asyncio.Event) -> None:
        """Run an interruptible worker loop until pool shutdown is requested."""
        while not stop.is_set():
            worked = await self.run_once()
            if worked:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self._policy.idle_poll_seconds)

    async def shutdown(self) -> None:
        """Detach active observation without inventing a provider terminal outcome."""
        context = self._active_context
        observation = self._active_observation
        active_run = self._active_run
        if context is not None:
            await context.detach(RuntimeShutdown())
            await context.wait_for_detach_delivery(min(self._policy.idle_poll_seconds, 0.1))
        if observation is not None and not observation.done():
            observation.cancel()
        elif (
            context is None
            and active_run is not None
            and active_run is not asyncio.current_task()
            and not active_run.done()
        ):
            active_run.cancel()

    async def _prepare_and_observe(
        self,
        claimed: ClaimedJob,
        lease_lost: asyncio.Event,
        context_holder: list[StoreBackedExecutionContext],
        observation_holder: list[asyncio.Task[None]],
    ) -> None:
        token = claimed.token
        context: StoreBackedExecutionContext | None = None
        try:
            if claimed.job.state == "queued":
                await self._store.mark_running(
                    token,
                    (),
                    event=BackendEvent(type="job_started", payload={"state": "running"}),
                )
                self._notifier.notify()
            if lease_lost.is_set():
                return

            workspace = await self._store.resolve_workspace(
                WorkspaceSelector(workspace_id=claimed.job.workspace_id)
            )
            backend = self._backends.get(claimed.job.backend_id)
            availability = await backend.check_availability(workspace)
            if lease_lost.is_set():
                return
            if availability.authenticated is False:
                await self._terminalize_error(
                    token,
                    JobError(
                        code="authentication_required",
                        message=availability.reason or "Backend authentication is required",
                    ),
                )
                return
            if not availability.available:
                await self._terminalize_error(
                    token,
                    JobError(
                        code="backend_unavailable",
                        message=availability.reason or "Backend is unavailable",
                        recoverable=True,
                    ),
                )
                return

            resolved_config = claimed.job.resolved_config
            if resolved_config is None:
                resolved_config = await backend.resolve_execution_config(
                    claimed.job.requested_config,
                    workspace,
                )
                if lease_lost.is_set():
                    return
                await self._store.store_resolved_config(token, resolved_config)
            current_attempts = await self._store.get_job_attempts(claimed.job.job_id)
            current_job = await self._store.get_job(claimed.job.job_id)
            if current_job is None:
                return
            context = StoreBackedExecutionContext(
                store=self._store,
                notifier=self._notifier,
                token=token,
                job=current_job,
                attempt=current_attempts[-1],
                workspace=workspace,
                resolved_config=resolved_config,
                control_poll_seconds=self._policy.idle_poll_seconds,
                output_chunk_bytes=self._output_chunk_bytes,
            )
            context_holder.append(context)
            self._active_context = context
            if lease_lost.is_set():
                await context.detach(LeaseLost())
                return
            observation = asyncio.create_task(self._observe(claimed, backend, context))
            observation_holder.append(observation)
            self._active_observation = observation
            if lease_lost.is_set():
                await context.detach(LeaseLost())
                observation.cancel()
            try:
                await observation
            except asyncio.CancelledError:
                if context.detached_signal is not None:
                    return
                if context.cancel_observed or await self._cancel_is_persisted(token):
                    await self._terminalize_cancelled(token, context=context)
                    return
                raise
        except StaleLeaseError:
            return
        except asyncio.CancelledError:
            raise
        except BackendFailure as failure:
            if context is not None:
                raise
            await self._handle_preparation_failure(claimed, failure)
        except NexusCoreError as error:
            if context is not None:
                raise
            try:
                code = _JOB_ERROR_CODE_ADAPTER.validate_python(error.code)
            except ValidationError:
                code = "internal_error"
            message = str(error) if code != "internal_error" else "Backend preparation failed"
            await self._terminalize_preparation_error(
                token,
                JobError(code=code, message=message),
            )
        except Exception:
            if context is not None:
                raise
            await self._terminalize_preparation_error(
                token,
                JobError(
                    code="internal_error",
                    message="Backend preparation failed",
                ),
            )

    async def _handle_preparation_failure(
        self,
        claimed: ClaimedJob,
        failure: BackendFailure,
    ) -> None:
        """Apply an explicit backend classification before provider execution begins."""
        error = failure.error.model_copy(update={"retry_disposition": failure.retry_disposition})
        try:
            if failure.retry_disposition == "reconcile_required":
                await self._store.mark_reconciling(
                    claimed.token,
                    error,
                    event=BackendEvent(
                        type="reconciliation",
                        payload={"status": "required"},
                    ),
                )
                self._notifier.notify()
                return
            if failure.retry_disposition == "safe_to_retry":
                retry_policy = (
                    claimed.job.resolved_config.retry_policy
                    if claimed.job.resolved_config is not None
                    else claimed.job.requested_config.explicit.retry_policy
                ) or RetryPolicy()
                if claimed.attempt.attempt_number < retry_policy.max_attempts:
                    delay = self._retry_delay(
                        claimed.attempt.attempt_number,
                        _retry_after_seconds(error),
                        retry_policy,
                    )
                    await self._store.schedule_retry(
                        claimed.token,
                        self._now() + timedelta(seconds=delay),
                        error,
                        event=BackendEvent(
                            type="retry_scheduled",
                            payload={"attempt": claimed.attempt.attempt_number + 1},
                        ),
                    )
                    self._notifier.notify()
                    return
            await self._terminalize_error(claimed.token, error)
        except StaleLeaseError:
            return

    async def _terminalize_preparation_error(
        self,
        token: LeaseToken,
        error: JobError,
    ) -> None:
        """Commit a pre-context error only while the original claim remains current."""
        try:
            await self._terminalize_error(token, error)
        except StaleLeaseError:
            return

    async def _observe(
        self,
        claimed: ClaimedJob,
        backend: AgentBackend,
        context: StoreBackedExecutionContext,
    ) -> None:
        attempts = await self._store.get_job_attempts(claimed.job.job_id)
        references = await self._store.get_provider_references(job_id=claimed.job.job_id)
        must_reconcile = any(
            attempt.reconciliation_classification is not None for attempt in attempts[:-1]
        ) or (
            claimed.attempt.phase == "reconciling"
            and not (len(attempts) >= 2 and attempts[-2].retry_classification == "safe_to_retry")
        )
        if must_reconcile:
            await self._reconcile(claimed, backend, context, references)
            return
        try:
            timeout_seconds = context.resolved_config.timeout_seconds
            if timeout_seconds is None:
                result = await backend.execute(claimed.job.operation, context)
            else:
                async with asyncio.timeout(timeout_seconds):
                    result = await backend.execute(
                        claimed.job.operation,
                        context,
                    )
        except BackendFailure as failure:
            await self._handle_execution_failure(claimed, backend, context, failure)
        except TimeoutError:
            await self._handle_execution_timeout(claimed, backend, context)
        except asyncio.CancelledError:
            raise
        except StaleLeaseError:
            return
        except Exception:
            await self._handle_unexpected_execution(claimed, backend, context)
        else:
            await self._terminalize_success(claimed.token, context, result)

    async def _handle_execution_failure(
        self,
        claimed: ClaimedJob,
        backend: AgentBackend,
        context: StoreBackedExecutionContext,
        failure: BackendFailure,
    ) -> None:
        error = failure.error.model_copy(update={"retry_disposition": failure.retry_disposition})
        references = await self._store.get_provider_references(job_id=claimed.job.job_id)
        if failure.retry_disposition == "reconcile_required" or references:
            await self._begin_reconciliation(claimed, backend, context, references, error)
            return
        if failure.retry_disposition == "safe_to_retry":
            retry_policy = context.resolved_config.retry_policy or RetryPolicy()
            if claimed.attempt.attempt_number < retry_policy.max_attempts:
                retry_after = _retry_after_seconds(error)
                delay = self._retry_delay(
                    claimed.attempt.attempt_number,
                    retry_after,
                    retry_policy,
                )
                await context.flush_output()
                await self._store.schedule_retry(
                    claimed.token,
                    self._now() + timedelta(seconds=delay),
                    error,
                    event=BackendEvent(
                        type="retry_scheduled",
                        payload={"attempt": claimed.attempt.attempt_number + 1},
                    ),
                )
                self._notifier.notify()
                return
        await self._terminalize_error(claimed.token, error, context=context)

    async def _handle_execution_timeout(
        self,
        claimed: ClaimedJob,
        backend: AgentBackend,
        context: StoreBackedExecutionContext,
    ) -> None:
        references = await self._store.get_provider_references(job_id=claimed.job.job_id)
        if references:
            error = JobError(
                code="process_lost",
                message="Provider completion is uncertain after timeout",
                retry_disposition="reconcile_required",
                recoverable=True,
            )
            await self._begin_reconciliation(claimed, backend, context, references, error)
            return
        await self._terminalize_error(
            claimed.token,
            JobError(code="timeout", message="Backend execution timed out"),
            context=context,
        )

    async def _handle_unexpected_execution(
        self,
        claimed: ClaimedJob,
        backend: AgentBackend,
        context: StoreBackedExecutionContext,
    ) -> None:
        references = await self._store.get_provider_references(job_id=claimed.job.job_id)
        if references:
            error = JobError(
                code="outcome_unknown",
                message="Provider completion is uncertain after an internal observation error",
                retry_disposition="reconcile_required",
            )
            await self._begin_reconciliation(claimed, backend, context, references, error)
            return
        await self._terminalize_error(
            claimed.token,
            JobError(code="internal_error", message="Unexpected backend execution failure"),
            context=context,
        )

    async def _begin_reconciliation(
        self,
        claimed: ClaimedJob,
        backend: AgentBackend,
        context: StoreBackedExecutionContext,
        references: tuple[ProviderReference, ...],
        error: JobError,
    ) -> None:
        await context.flush_output()
        reconcile_error = error.model_copy(update={"retry_disposition": "reconcile_required"})
        await self._store.mark_reconciling(
            claimed.token,
            reconcile_error,
            event=BackendEvent(
                type="reconciliation",
                payload={"status": "required"},
            ),
        )
        self._notifier.notify()
        await self._reconcile(claimed, backend, context, references)

    async def _reconcile(
        self,
        claimed: ClaimedJob,
        backend: AgentBackend,
        context: StoreBackedExecutionContext,
        references: tuple[ProviderReference, ...],
    ) -> None:
        try:
            async with asyncio.timeout(self._policy.reconciliation_timeout_seconds):
                outcome = await backend.reconcile(references, context)
        except BackendFailure as failure:
            if failure.retry_disposition == "terminal":
                await self._terminalize_error(claimed.token, failure.error, context=context)
                return
            await self._store.mark_reconciling(
                claimed.token,
                failure.error.model_copy(update={"retry_disposition": "reconcile_required"}),
                event=BackendEvent(type="reconciliation", payload={"status": "deferred"}),
            )
            self._notifier.notify()
            return
        except TimeoutError:
            await self._store.mark_reconciling(
                claimed.token,
                JobError(
                    code="outcome_unknown",
                    message="Provider reconciliation timed out",
                    retry_disposition="reconcile_required",
                ),
                event=BackendEvent(type="reconciliation", payload={"status": "timeout"}),
            )
            self._notifier.notify()
            return
        except asyncio.CancelledError:
            raise
        except StaleLeaseError:
            return
        except Exception:
            await self._store.mark_reconciling(
                claimed.token,
                JobError(
                    code="outcome_unknown",
                    message="Provider reconciliation could not determine an outcome",
                    retry_disposition="reconcile_required",
                ),
                event=BackendEvent(type="reconciliation", payload={"status": "deferred"}),
            )
            self._notifier.notify()
            return

        match outcome:
            case CompletedReconciliationOutcome(result=result):
                await self._terminalize_success(claimed.token, context, result)
            case FailedReconciliationOutcome(error=error):
                await self._terminalize_error(claimed.token, error, context=context)
            case CancelledReconciliationOutcome():
                await self._terminalize_cancelled(claimed.token, context=context)
            case UnknownReconciliationOutcome(error=error):
                await self._terminalize_error(claimed.token, error, context=context)
            case ActiveReconciliationOutcome() | InputRequiredReconciliationOutcome():
                await context.flush_output()

    async def _terminalize_success(
        self,
        token: LeaseToken,
        context: StoreBackedExecutionContext,
        result: OperationResult,
    ) -> None:
        if result.kind != context.job.operation.kind:
            await self._terminalize_error(
                token,
                JobError(
                    code="internal_error",
                    message="Backend result kind does not match the admitted operation",
                ),
                context=context,
            )
            return
        await context.flush_output()
        if isinstance(result, TurnResult):
            await context.emit(
                BackendEvent(
                    type="message",
                    payload={"text": result.message, "final": True},
                )
            )
        await self._store.terminalize(
            token,
            SucceededTerminalOutcome(result=result),
            event=BackendEvent(type="job_completed"),
        )
        self._notifier.notify()

    async def _terminalize_error(
        self,
        token: LeaseToken,
        error: JobError,
        *,
        context: StoreBackedExecutionContext | None = None,
    ) -> None:
        if context is not None:
            await context.flush_output()
        await self._store.terminalize(
            token,
            FailedTerminalOutcome(error=error),
            event=BackendEvent(type="job_failed", payload={"code": error.code}),
        )
        self._notifier.notify()

    async def _terminalize_cancelled(
        self,
        token: LeaseToken,
        *,
        context: StoreBackedExecutionContext | None = None,
    ) -> None:
        if context is not None:
            await context.flush_output()
        await self._store.terminalize(
            token,
            CancelledTerminalOutcome(),
            event=BackendEvent(type="job_cancelled"),
        )
        self._notifier.notify()

    async def _cancel_is_persisted(self, token: LeaseToken) -> bool:
        try:
            return (await self._store.get_control_snapshot(token)).cancel_requested
        except StaleLeaseError:
            return False

    async def _heartbeat(
        self,
        token: LeaseToken,
        stop: asyncio.Event,
        lease_lost: asyncio.Event,
        context_holder: list[StoreBackedExecutionContext],
        observation_holder: list[asyncio.Task[None]],
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._policy.heartbeat_seconds)
                return
            except TimeoutError:
                pass
            renewed = await self._renew_lease_interruptibly(token, stop)
            if renewed is None:
                return
            if renewed:
                continue
            lease_lost.set()
            if context_holder:
                await context_holder[0].detach(LeaseLost())
                await context_holder[0].wait_for_detach_delivery(self._policy.heartbeat_seconds)
            if observation_holder and not observation_holder[0].done():
                observation_holder[0].cancel()
            return

    async def _renew_lease_interruptibly(
        self,
        token: LeaseToken,
        stop: asyncio.Event,
    ) -> bool | None:
        """Renew once, or cancel and await the renewal when worker shutdown wins."""
        renewal_task = asyncio.create_task(
            self._store.renew_lease(
                token,
                self._now() + timedelta(seconds=self._policy.lease_seconds),
            )
        )
        stop_task = asyncio.create_task(stop.wait())
        tasks = (renewal_task, stop_task)
        try:
            completed, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if stop_task in completed:
                return None
            if renewal_task.cancelled():
                return False
            try:
                return await renewal_task
            except Exception:
                return False
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("worker clock must return timezone-aware values")
        return now.astimezone(UTC)


class WorkerPool:
    """Own a bounded set of interruptible job worker loops."""

    def __init__(
        self,
        *,
        store: JobStore,
        backends: BackendManager,
        notifier: EventNotifier,
        worker_count: int = 1,
        worker_id_prefix: str = "nexus-worker",
        policy: WorkerPolicy | None = None,
        retry_delay: RetryDelay | None = None,
        clock: Clock | None = None,
        output_chunk_bytes: int = 4096,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        self._stop = asyncio.Event()
        self._workers = tuple(
            JobWorker(
                worker_id=f"{worker_id_prefix}-{index + 1}",
                store=store,
                backends=backends,
                notifier=notifier,
                policy=policy,
                retry_delay=retry_delay,
                clock=clock,
                output_chunk_bytes=output_chunk_bytes,
            )
            for index in range(worker_count)
        )
        self._tasks: tuple[asyncio.Task[None], ...] = ()

    @property
    def running(self) -> bool:
        """Return whether any owned background worker loop remains active."""
        return any(not task.done() for task in self._tasks)

    async def start(self) -> None:
        """Start every configured worker loop exactly once per lifecycle."""
        if self.running:
            return
        self._stop = asyncio.Event()
        self._tasks = tuple(asyncio.create_task(worker.run(self._stop)) for worker in self._workers)

    async def stop(self) -> None:
        """Signal and await every loop without waiting for an idle poll deadline."""
        self._stop.set()
        await asyncio.gather(*(worker.shutdown() for worker in self._workers))
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = ()

    async def run_until_idle(self) -> int:
        """Drain currently eligible jobs deterministically without background loops."""
        if self.running:
            raise RuntimeError("run_until_idle cannot run while the pool is started")
        completed = 0
        while True:
            progress = False
            for worker in self._workers:
                if await worker.run_once():
                    completed += 1
                    progress = True
            if not progress:
                return completed


def _retry_after_seconds(error: JobError) -> float | None:
    value = error.details.get("retry_after_seconds")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return max(0.0, float(value))
