"""Shared durable job-store and application-service contract fixtures."""

from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from nexus_mcp.backends import BackendManager
from nexus_mcp.core import (
    BackendEvent,
    CancelReceipt,
    DiagnosticsOperation,
    InputResolutionReceipt,
    ProviderReference,
    RequestedExecutionConfig,
)
from nexus_mcp.jobs.configuration import NexusConfigResolver
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.service import AgentJobService
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore
from nexus_mcp.jobs.store import (
    CancelJobCommand,
    CreateJobCommand,
    CreateJobResult,
    EventPage,
    JobStore,
    StoredJobPage,
)
from tests.fixtures import (
    make_agent_job,
    make_agent_session,
    make_backend_descriptor,
    make_job_handle,
    make_workspace,
)
from tests.job_fakes import InMemoryJobStore, ScriptedBackend
from tests.unit.jobs._service_support import NOW, _ChangingCaptureResolver


@pytest.fixture(params=["memory", "sqlite"], ids=["memory", "sqlite"])
async def admission_store(request: pytest.FixtureRequest, tmp_path: Path):
    """Run admission, identity, and query assertions against every implementation."""
    store = (
        SQLiteJobStore(tmp_path / "jobs.sqlite3")
        if request.param == "sqlite"
        else InMemoryJobStore()
    )
    await store.open()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture(params=["memory", "sqlite"], ids=["memory", "sqlite"])
async def job_store(request: pytest.FixtureRequest, tmp_path: Path):
    """Run the complete lifecycle contract against every store implementation."""
    store = (
        SQLiteJobStore(tmp_path / "jobs.sqlite3")
        if request.param == "sqlite"
        else InMemoryJobStore()
    )
    await store.open()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
def store(tmp_path: Path) -> Mock:
    """Provide a store double with complete, representative service responses."""
    job_store = Mock(spec=JobStore)
    workspace = make_workspace(canonical_path=tmp_path)
    job_store.resolve_workspace.return_value = workspace
    job_store.resolve_or_create_workspace.return_value = workspace
    job_store.get_session.return_value = make_agent_session()
    job_store.get_provider_references.return_value = (
        ProviderReference(kind="thread", value="thread-test"),
    )
    job_store.get_job.return_value = make_agent_job()
    job_store.get_job_attempts.return_value = ()
    job_store.get_pending_inputs.return_value = ()
    job_store.get_job_result.return_value = None
    job_store.read_events.return_value = EventPage()
    job_store.list_jobs.return_value = StoredJobPage()
    job_store.request_cancel.return_value = CancelReceipt(
        job_id="job-test",
        state="cancelled",
        cancel_requested=True,
        completed_immediately=True,
        event_committed=True,
    )
    job_store.resolve_input.return_value = InputResolutionReceipt(
        job_id="job-test", input_id="input-test"
    )

    async def create_job(command):
        return CreateJobResult(
            handle=make_job_handle(
                job_id="job-created",
                session_id=command.session_id,
                operation=command.operation,
            ),
            created=True,
        )

    job_store.create_job.side_effect = create_job
    return job_store


@pytest.fixture
def backend() -> Mock:
    """Provide one fully capable registered backend double."""
    registered = Mock()
    registered.descriptor = make_backend_descriptor()
    return registered


@pytest.fixture
def manager(backend: Mock) -> Mock:
    """Provide deterministic static capability lookup without health calls."""
    backend_manager = Mock(spec=BackendManager)
    backend_manager.get.return_value = backend
    backend_manager.require_operation.return_value = backend
    backend_manager.list_statuses.return_value = ()
    return backend_manager


@pytest.fixture
def config_resolver() -> Mock:
    """Snapshot only the explicit layer supplied by each service test."""
    resolver = Mock(spec=NexusConfigResolver)
    resolver.snapshot.side_effect = lambda backend_id, workspace, explicit: (
        RequestedExecutionConfig(explicit=explicit)
    )
    return resolver


@pytest.fixture
def notifier() -> EventNotifier:
    """Expose local post-commit wake revisions to tests."""
    return EventNotifier()


@pytest.fixture
def service(
    store: Mock,
    manager: Mock,
    config_resolver: Mock,
    notifier: EventNotifier,
) -> AgentJobService:
    """Construct the framework-independent service over boundary doubles."""
    return AgentJobService(
        store=store,
        backend_manager=manager,
        config_resolver=config_resolver,
        notifier=notifier,
    )


@pytest.fixture(params=["memory", "sqlite"], ids=["memory", "sqlite"])
async def real_service_environment(
    request: pytest.FixtureRequest,
    tmp_path: Path,
):
    """Run application idempotency behavior against both complete store implementations."""
    durable_store: JobStore = (
        SQLiteJobStore(tmp_path / "service.sqlite3")
        if request.param == "sqlite"
        else InMemoryJobStore()
    )
    await durable_store.open()
    workspace = make_workspace(canonical_path=tmp_path, created_at=NOW, updated_at=NOW)
    seed = await durable_store.create_job(
        CreateJobCommand(
            workspace=workspace,
            backend_id="codex",
            owner_id="local:501",
            access_policy="private",
            operation=DiagnosticsOperation(),
            requested_config=RequestedExecutionConfig(),
            session_id=None,
            create_session=False,
            command_family="seed",
            queued_event=BackendEvent(type="job_queued", occurred_at=NOW),
        )
    )
    await durable_store.request_cancel(
        CancelJobCommand(
            job_id=seed.handle.job_id,
            requested_at=NOW,
            active_cancellation_allowed=False,
            queued_event=BackendEvent(type="job_cancelled", occurred_at=NOW),
            active_event=BackendEvent(type="cancel_requested", occurred_at=NOW),
        )
    )
    backend = ScriptedBackend(backend_id="codex")
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(
                update={
                    "sandbox_modes": frozenset(
                        {"read_only", "workspace_write", "danger_full_access"}
                    ),
                    "review_targets": frozenset(
                        {"working_tree", "branch", "commit", "pull_request"}
                    ),
                    "review_deliveries": frozenset({"inline", "detached"}),
                }
            )
        }
    )
    notifier = EventNotifier()
    service = AgentJobService(
        store=durable_store,
        backend_manager=BackendManager([backend]),
        config_resolver=cast("NexusConfigResolver", _ChangingCaptureResolver()),
        notifier=notifier,
    )
    try:
        yield service, durable_store, backend, notifier
    finally:
        await durable_store.close()
