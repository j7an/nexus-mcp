"""Canonical framework-independent application service for durable agent jobs."""

import asyncio
from collections.abc import Awaitable, Callable

from nexus_mcp.backends import AgentBackend, BackendManager
from nexus_mcp.core import (
    TERMINAL_STATES,
    AccessContext,
    AccessDeniedError,
    AccessPolicy,
    AgentJob,
    AgentOperation,
    AgentSession,
    BackendEvent,
    BackendStatus,
    CancelledJobResultResponse,
    CancelReceipt,
    DiagnosticsOperation,
    ExecutionConfigValues,
    FailedJobResultResponse,
    ForkOperation,
    InputResolutionReceipt,
    InputResponse,
    JobError,
    JobEvent,
    JobEventType,
    JobHandle,
    JobListPage,
    JobNotFoundError,
    JobResultEnvelope,
    JobResultResponse,
    JobState,
    JobStatus,
    PendingJobResultResponse,
    ProviderReference,
    RequestedExecutionConfig,
    ReviewOperation,
    SessionNotFoundError,
    SucceededJobResultResponse,
    TurnOperation,
    UnsupportedCapabilityError,
    Workspace,
    WorkspaceInvalidError,
    WorkspaceSelector,
    new_id,
)
from nexus_mcp.jobs.configuration import NexusConfigResolver
from nexus_mcp.jobs.events import EventNotifier, JobEventSubscription
from nexus_mcp.jobs.store import (
    CancelJobCommand,
    CreateJobCommand,
    JobAccessFilter,
    JobQuery,
    JobStore,
    ResolveInputCommand,
)

__all__ = ["AgentJobService"]

_ALL_JOB_STATES: frozenset[JobState] = frozenset(
    {"queued", "running", "input_required", "completed", "failed", "cancelled"}
)
_MUTATING_SANDBOXES = frozenset({"workspace_write", "danger_full_access"})


class _AuthorizedJobEventSubscription(JobEventSubscription):
    """Authorize one synchronous subscription before its first asynchronous store read."""

    def __init__(
        self,
        store: JobStore,
        notifier: EventNotifier,
        job_id: str,
        after: int,
        authorize: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(store, notifier, job_id, after)
        self._authorize = authorize
        self._authorization_complete = False

    async def __anext__(self) -> JobEvent:
        """Hide event history until the durable job passes async access checks."""
        if not self._authorization_complete:
            await self._authorize()
            self._authorization_complete = True
        return await super().__anext__()


class AgentJobService:
    """Admit, query, and control durable jobs without a transport-framework dependency."""

    def __init__(
        self,
        *,
        store: JobStore,
        backend_manager: BackendManager,
        config_resolver: NexusConfigResolver,
        notifier: EventNotifier,
    ) -> None:
        self._store = store
        self._backend_manager = backend_manager
        self._config_resolver = config_resolver
        self._notifier = notifier

    async def start(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        backend_id: str,
        operation: TurnOperation,
        explicit_config: ExecutionConfigValues,
        access_policy: AccessPolicy = "private",
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Create a new session and queue its initial turn."""
        resolved_workspace = await self._resolve_creation_workspace(workspace, access)
        if access_policy == "workspace" and not self._workspace_authorized(
            access, resolved_workspace
        ):
            raise AccessDeniedError("Workspace policy requires workspace authorization")
        backend = self._require_backend(backend_id, operation, explicit_config)
        requested_config = self._snapshot_config(backend, resolved_workspace, explicit_config)
        return await self._create(
            workspace=resolved_workspace,
            backend_id=backend_id,
            owner_id=access.principal_id,
            access_policy=access_policy,
            operation=operation,
            requested_config=requested_config,
            session_id=new_id(),
            create_session=True,
            command_family="start",
            idempotency_key=idempotency_key,
        )

    async def continue_session(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        session_id: str,
        operation: TurnOperation,
        explicit_config: ExecutionConfigValues,
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Queue a turn against one authorized existing session."""
        resolved_workspace, session = await self._resolve_creation_session(
            workspace, access, session_id
        )
        backend = self._require_backend(session.backend_id, operation, explicit_config)
        checkpoint = await self._store.get_provider_references(session_id=session.session_id)
        requested_config = self._snapshot_config(backend, resolved_workspace, explicit_config)
        return await self._create(
            workspace=resolved_workspace,
            backend_id=session.backend_id,
            owner_id=session.owner_id,
            access_policy=session.access_policy,
            operation=operation,
            requested_config=requested_config,
            session_id=session.session_id,
            create_session=False,
            source_checkpoint=checkpoint,
            command_family="continue_session",
            idempotency_key=idempotency_key,
        )

    async def fork_session(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        session_id: str,
        operation: ForkOperation,
        explicit_config: ExecutionConfigValues,
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Create a child session from an authorized parent checkpoint."""
        resolved_workspace, session = await self._resolve_creation_session(
            workspace, access, session_id
        )
        backend = self._require_backend(session.backend_id, operation, explicit_config)
        self._require_session_fork(backend)
        checkpoint = await self._store.get_provider_references(session_id=session.session_id)
        requested_config = self._snapshot_config(backend, resolved_workspace, explicit_config)
        return await self._create(
            workspace=resolved_workspace,
            backend_id=session.backend_id,
            owner_id=session.owner_id,
            access_policy=session.access_policy,
            operation=operation,
            requested_config=requested_config,
            session_id=new_id(),
            create_session=True,
            parent_session_id=session.session_id,
            source_checkpoint=checkpoint,
            command_family="fork_session",
            idempotency_key=idempotency_key,
        )

    async def review(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        session_id: str,
        operation: ReviewOperation,
        explicit_config: ExecutionConfigValues,
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Queue an inline review or a detached review child session."""
        resolved_workspace, session = await self._resolve_creation_session(
            workspace, access, session_id
        )
        backend = self._require_backend(session.backend_id, operation, explicit_config)
        self._require_review_capabilities(backend, operation)
        detached = operation.delivery == "detached"
        if detached:
            self._require_session_fork(backend)
        checkpoint = await self._store.get_provider_references(session_id=session.session_id)
        requested_config = self._snapshot_config(backend, resolved_workspace, explicit_config)
        return await self._create(
            workspace=resolved_workspace,
            backend_id=session.backend_id,
            owner_id=session.owner_id,
            access_policy=session.access_policy,
            operation=operation,
            requested_config=requested_config,
            session_id=new_id() if detached else session.session_id,
            create_session=detached,
            parent_session_id=session.session_id if detached else None,
            source_checkpoint=checkpoint,
            command_family="review",
            idempotency_key=idempotency_key,
        )

    async def diagnose(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        backend_id: str,
        explicit_config: ExecutionConfigValues,
        idempotency_key: str | None = None,
    ) -> JobHandle:
        """Queue sessionless backend diagnostics without a health preflight."""
        resolved_workspace = await self._resolve_creation_workspace(workspace, access)
        operation = DiagnosticsOperation()
        backend = self._require_backend(backend_id, operation, explicit_config)
        requested_config = self._snapshot_config(backend, resolved_workspace, explicit_config)
        return await self._create(
            workspace=resolved_workspace,
            backend_id=backend_id,
            owner_id=access.principal_id,
            access_policy="private",
            operation=operation,
            requested_config=requested_config,
            session_id=None,
            create_session=False,
            command_family="diagnose",
            idempotency_key=idempotency_key,
        )

    async def status(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        job_id: str,
    ) -> JobStatus:
        """Return one complete authorized durable status projection."""
        resolved_workspace = await self._store.resolve_workspace(workspace)
        job = await self._authorized_job(resolved_workspace, access, job_id)
        return await self._status_from_job(job)

    async def result(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        job_id: str,
    ) -> JobResultResponse:
        """Return the strict public result variant implied by durable job state."""
        resolved_workspace = await self._store.resolve_workspace(workspace)
        job = await self._authorized_job(resolved_workspace, access, job_id)
        match job.state:
            case "queued" | "running" | "input_required":
                return PendingJobResultResponse(job_id=job.job_id, state=job.state)
            case "completed":
                stored = await self._store.get_job_result(job.job_id)
                if not isinstance(stored, JobResultEnvelope):
                    raise RuntimeError("completed job is missing its durable result")
                return SucceededJobResultResponse(job_id=job.job_id, result=stored)
            case "failed":
                stored = await self._store.get_job_result(job.job_id)
                if not isinstance(stored, JobError):
                    raise RuntimeError("failed job is missing its durable error")
                return FailedJobResultResponse(job_id=job.job_id, error=stored)
            case "cancelled":
                return CancelledJobResultResponse(job_id=job.job_id)

    async def cancel(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        job_id: str,
    ) -> CancelReceipt:
        """Request idempotent queued or provider-supported active cancellation."""
        resolved_workspace = await self._store.resolve_workspace(workspace)
        job = await self._authorized_job(resolved_workspace, access, job_id)
        records_event = job.state not in TERMINAL_STATES and job.cancel_requested_at is None
        if job.state in {"running", "input_required"} and records_event:
            backend = self._backend_manager.get(job.backend_id)
            if not backend.descriptor.capabilities.cancellation:
                raise UnsupportedCapabilityError(job.backend_id, "cancellation")
        event_type: JobEventType = "job_cancelled" if job.state == "queued" else "cancel_requested"
        receipt = await self._store.request_cancel(
            CancelJobCommand(
                job_id=job.job_id,
                event=BackendEvent(type=event_type, payload={"state": job.state}),
            )
        )
        if records_event:
            self._notifier.notify()
        return receipt

    async def respond(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        job_id: str,
        input_id: str,
        response: InputResponse,
    ) -> InputResolutionReceipt:
        """Persist one authorized, validated, idempotent input response."""
        resolved_workspace = await self._store.resolve_workspace(workspace)
        job = await self._authorized_job(resolved_workspace, access, job_id)
        receipt = await self._store.resolve_input(
            ResolveInputCommand(
                job_id=job.job_id,
                input_id=input_id,
                response=response,
                event=BackendEvent(type="input_resolved", payload={"input_id": input_id}),
            )
        )
        if not receipt.replayed:
            self._notifier.notify()
        return receipt

    async def list_jobs(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        states: frozenset[JobState] = frozenset(),
        limit: int = 50,
        cursor: str | None = None,
    ) -> JobListPage:
        """Return one authorized, bounded page of complete status projections."""
        resolved_workspace = await self._store.resolve_workspace(workspace)
        workspace_authorized = self._workspace_authorized(access, resolved_workspace)
        page = await self._store.list_jobs(
            JobQuery(
                workspace_id=resolved_workspace.workspace_id,
                access=JobAccessFilter(
                    principal_id=access.principal_id,
                    workspace_authorized=workspace_authorized,
                ),
                states=states or _ALL_JOB_STATES,
                limit=limit,
                cursor=cursor,
            )
        )
        visible_jobs = tuple(
            job
            for job in page.jobs
            if job.workspace_id == resolved_workspace.workspace_id
            and self._job_visible(job, resolved_workspace, access)
        )
        items = await asyncio.gather(*(self._status_from_job(job) for job in visible_jobs))
        return JobListPage(items=tuple(items), next_cursor=page.next_cursor)

    def subscribe_events(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
        job_id: str,
        after_sequence: int = 0,
    ) -> JobEventSubscription:
        """Create a stream whose first iteration authorizes before reading events."""

        async def authorize() -> None:
            resolved_workspace = await self._store.resolve_workspace(workspace)
            await self._authorized_job(resolved_workspace, access, job_id)

        return _AuthorizedJobEventSubscription(
            self._store,
            self._notifier,
            job_id,
            after_sequence,
            authorize,
        )

    async def list_backends(
        self,
        *,
        workspace: WorkspaceSelector,
        access: AccessContext,
    ) -> tuple[BackendStatus, ...]:
        """Return fresh backend health only to a workspace-authorized caller."""
        resolved_workspace = await self._store.resolve_workspace(workspace)
        self._require_workspace_access(access, resolved_workspace)
        return await self._backend_manager.list_statuses(resolved_workspace)

    async def _create(
        self,
        *,
        workspace: Workspace,
        backend_id: str,
        owner_id: str,
        access_policy: AccessPolicy,
        operation: AgentOperation,
        requested_config: RequestedExecutionConfig,
        session_id: str | None,
        create_session: bool,
        command_family: str,
        idempotency_key: str | None,
        parent_session_id: str | None = None,
        source_checkpoint: tuple[ProviderReference, ...] = (),
    ) -> JobHandle:
        result = await self._store.create_job(
            CreateJobCommand(
                workspace=workspace,
                backend_id=backend_id,
                owner_id=owner_id,
                access_policy=access_policy,
                operation=operation,
                requested_config=requested_config,
                session_id=session_id,
                create_session=create_session,
                parent_session_id=parent_session_id,
                source_checkpoint=source_checkpoint,
                command_family=command_family,
                idempotency_key=idempotency_key,
                queued_event=BackendEvent(type="job_queued", payload={"state": "queued"}),
            )
        )
        if result.created:
            self._notifier.notify()
        return result.handle

    async def _resolve_creation_workspace(
        self,
        selector: WorkspaceSelector,
        access: AccessContext,
    ) -> Workspace:
        workspace = await self._store.resolve_workspace(selector)
        path = workspace.canonical_path
        if not path.exists():
            raise WorkspaceInvalidError(str(path), "workspace path does not exist")
        if not path.is_dir():
            raise WorkspaceInvalidError(str(path), "workspace path is not a directory")
        self._require_workspace_access(access, workspace)
        return workspace

    async def _resolve_creation_session(
        self,
        selector: WorkspaceSelector,
        access: AccessContext,
        session_id: str,
    ) -> tuple[Workspace, AgentSession]:
        workspace = await self._resolve_creation_workspace(selector, access)
        session = await self._store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        if session.workspace_id != workspace.workspace_id or not self._session_visible(
            session, workspace, access
        ):
            raise AccessDeniedError("Session access denied")
        return workspace, session

    async def _authorized_job(
        self,
        workspace: Workspace,
        access: AccessContext,
        job_id: str,
    ) -> AgentJob:
        job = await self._store.get_job(job_id)
        if (
            job is None
            or job.workspace_id != workspace.workspace_id
            or not self._job_visible(job, workspace, access)
        ):
            raise JobNotFoundError(job_id)
        return job

    async def _status_from_job(self, job: AgentJob) -> JobStatus:
        attempts, pending_inputs, latest_sequence = await asyncio.gather(
            self._store.get_job_attempts(job.job_id),
            self._store.get_pending_inputs(job.job_id),
            self._latest_event_sequence(job.job_id),
        )
        phase = None if not attempts else attempts[-1].phase
        return JobStatus.from_job(
            job,
            phase=phase,
            pending_inputs=pending_inputs,
            latest_event_sequence=latest_sequence,
        )

    async def _latest_event_sequence(self, job_id: str) -> int:
        after_sequence = 0
        latest_sequence = 0
        while True:
            page = await self._store.read_events(job_id, after_sequence, 1000)
            if page.events:
                latest_sequence = page.events[-1].sequence
            if not page.has_more:
                return latest_sequence
            next_sequence = page.next_after_sequence
            if next_sequence is None or next_sequence <= after_sequence:
                raise RuntimeError("event page did not advance its durable cursor")
            after_sequence = next_sequence

    def _require_backend(
        self,
        backend_id: str,
        operation: AgentOperation,
        explicit_config: ExecutionConfigValues,
    ) -> AgentBackend:
        backend = self._backend_manager.require_operation(backend_id, operation.kind)
        self._validate_sandbox_modes(backend, (explicit_config,))
        self._validate_approval_policy(backend, explicit_config, (explicit_config,))
        return backend

    def _snapshot_config(
        self,
        backend: AgentBackend,
        workspace: Workspace,
        explicit_config: ExecutionConfigValues,
    ) -> RequestedExecutionConfig:
        requested = self._config_resolver.snapshot(
            backend.descriptor.backend_id,
            workspace,
            explicit_config,
        )
        layers = tuple(
            layer
            for layer in (
                requested.explicit,
                None if requested.workspace is None else requested.workspace.values,
                None if requested.user is None else requested.user.values,
                None if requested.environment is None else requested.environment.values,
            )
            if layer is not None
        )
        self._validate_sandbox_modes(backend, layers)
        self._validate_approval_policy(backend, explicit_config, layers)
        return requested

    @staticmethod
    def _validate_sandbox_modes(
        backend: AgentBackend,
        layers: tuple[ExecutionConfigValues, ...],
    ) -> None:
        advertised = backend.descriptor.capabilities.sandbox_modes
        for layer in layers:
            if layer.sandbox is not None and layer.sandbox not in advertised:
                raise UnsupportedCapabilityError(
                    backend.descriptor.backend_id,
                    f"sandbox:{layer.sandbox}",
                )

    @staticmethod
    def _validate_approval_policy(
        backend: AgentBackend,
        explicit_config: ExecutionConfigValues,
        layers: tuple[ExecutionConfigValues, ...],
    ) -> None:
        if not any(layer.approval_policy == "never" for layer in layers):
            return
        if explicit_config.sandbox not in _MUTATING_SANDBOXES:
            raise UnsupportedCapabilityError(
                backend.descriptor.backend_id,
                "approval_policy:never",
            )

    @staticmethod
    def _require_review_capabilities(
        backend: AgentBackend,
        operation: ReviewOperation,
    ) -> None:
        capabilities = backend.descriptor.capabilities
        if operation.delivery not in capabilities.review_deliveries:
            raise UnsupportedCapabilityError(
                backend.descriptor.backend_id,
                f"review_delivery:{operation.delivery}",
            )
        if operation.target.kind not in capabilities.review_targets:
            raise UnsupportedCapabilityError(
                backend.descriptor.backend_id,
                f"review_target:{operation.target.kind}",
            )

    @staticmethod
    def _require_session_fork(backend: AgentBackend) -> None:
        if not backend.descriptor.capabilities.session_fork:
            raise UnsupportedCapabilityError(
                backend.descriptor.backend_id,
                "session_fork",
            )

    @staticmethod
    def _workspace_authorized(access: AccessContext, workspace: Workspace) -> bool:
        return (
            access.authorize_local_workspaces
            or workspace.workspace_id in access.authorized_workspace_ids
        )

    def _require_workspace_access(
        self,
        access: AccessContext,
        workspace: Workspace,
    ) -> None:
        if not self._workspace_authorized(access, workspace):
            raise AccessDeniedError("Workspace access denied")

    def _session_visible(
        self,
        session: AgentSession,
        workspace: Workspace,
        access: AccessContext,
    ) -> bool:
        return session.owner_id == access.principal_id or (
            session.access_policy == "workspace" and self._workspace_authorized(access, workspace)
        )

    def _job_visible(
        self,
        job: AgentJob,
        workspace: Workspace,
        access: AccessContext,
    ) -> bool:
        return job.owner_id == access.principal_id or (
            job.access_policy == "workspace" and self._workspace_authorized(access, workspace)
        )
