"""Canonical application-service admission, access, query, and control behavior."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from nexus_mcp.backends import BackendManager
from nexus_mcp.core import (
    AccessDeniedError,
    BackendAvailability,
    BackendStatus,
    CancelledJobResultResponse,
    CancelReceipt,
    ExecutionConfigValues,
    FailedJobResultResponse,
    ForkOperation,
    InputAlreadyResolvedError,
    InputResolutionReceipt,
    JobAttempt,
    JobEvent,
    JobNotFoundError,
    JobResultEnvelope,
    PendingJobResultResponse,
    PermissionResponse,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    ReviewOperation,
    ReviewTarget,
    SucceededJobResultResponse,
    TurnOperation,
    UnsupportedCapabilityError,
    WorkspaceInvalidError,
    WorkspaceSelector,
)
from nexus_mcp.jobs.configuration import NexusConfigResolver
from nexus_mcp.jobs.events import EventNotifier, JobEventSubscription
from nexus_mcp.jobs.service import AgentJobService
from nexus_mcp.jobs.store import CreateJobResult, EventPage, JobStore, StoredJobPage
from tests.fixtures import (
    make_access_context,
    make_agent_job,
    make_agent_session,
    make_backend_descriptor,
    make_job_error,
    make_job_handle,
    make_pending_permission,
    make_turn_result,
    make_workspace,
)

NOW = datetime(2026, 8, 30, 20, 0, tzinfo=UTC)
WORKSPACE_SELECTOR = WorkspaceSelector(workspace_id="ws-test")
ALL_STATES = frozenset({"queued", "running", "input_required", "completed", "failed", "cancelled"})


def authorized_access(**overrides: Any):
    """Create a caller explicitly trusted for the representative workspace."""
    defaults = {"authorized_workspace_ids": frozenset({"ws-test"})}
    return make_access_context(**(defaults | overrides))


def make_review_operation(**overrides: Any) -> ReviewOperation:
    """Create a representative inline working-tree review."""
    defaults = {"target": ReviewTarget(kind="working_tree"), "delivery": "inline"}
    return ReviewOperation(**(defaults | overrides))


@pytest.fixture
def store(tmp_path: Path) -> Mock:
    """Provide a store double with complete, representative service responses."""
    job_store = Mock(spec=JobStore)
    workspace = make_workspace(canonical_path=tmp_path)
    job_store.resolve_workspace.return_value = workspace
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


async def test_start_commits_private_session_snapshot_before_notifying(
    service: AgentJobService,
    store: Mock,
    manager: Mock,
    notifier: EventNotifier,
):
    """Dropping atomic queued-event admission or post-commit wake ordering breaks new jobs."""

    async def create_job(command):
        assert notifier.revision == 0
        return CreateJobResult(
            handle=make_job_handle(
                job_id="job-created",
                session_id=command.session_id,
                operation=command.operation,
            ),
            created=True,
        )

    store.create_job.side_effect = create_job
    operation = TurnOperation(prompt="Inspect the workspace")

    handle = await service.start(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        backend_id="codex",
        operation=operation,
        explicit_config=ExecutionConfigValues(model="gpt-5"),
        idempotency_key="request-1",
    )

    command = store.create_job.await_args.args[0]
    assert handle.session_id == command.session_id
    assert command.create_session is True
    assert command.parent_session_id is None
    assert command.source_checkpoint == ()
    assert command.owner_id == "local:501"
    assert command.access_policy == "private"
    assert command.command_family == "start"
    assert command.idempotency_key == "request-1"
    assert command.queued_event.type == "job_queued"
    assert command.requested_config.explicit.model == "gpt-5"
    assert notifier.revision == 1
    manager.list_statuses.assert_not_awaited()


async def test_idempotent_start_replay_does_not_emit_a_false_wake(
    service: AgentJobService,
    store: Mock,
    notifier: EventNotifier,
):
    """A create replay commits no second queued event and must not wake as if it did."""
    store.create_job.side_effect = None
    store.create_job.return_value = CreateJobResult(handle=make_job_handle(), created=False)

    await service.start(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        backend_id="codex",
        operation=TurnOperation(prompt="Inspect"),
        explicit_config=ExecutionConfigValues(),
    )

    assert notifier.revision == 0


async def test_creation_requires_current_workspace_authorization(
    service: AgentJobService,
    store: Mock,
):
    """A trusted identity without workspace authority cannot launch filesystem work."""
    with pytest.raises(AccessDeniedError):
        await service.start(
            workspace=WORKSPACE_SELECTOR,
            access=make_access_context(),
            backend_id="codex",
            operation=TurnOperation(prompt="Inspect"),
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()


async def test_workspace_session_policy_requires_explicit_authorization(
    service: AgentJobService,
    store: Mock,
):
    """A new shared session cannot broaden visibility without workspace authority."""
    with pytest.raises(AccessDeniedError):
        await service.start(
            workspace=WORKSPACE_SELECTOR,
            access=make_access_context(),
            backend_id="codex",
            operation=TurnOperation(prompt="Inspect"),
            explicit_config=ExecutionConfigValues(),
            access_policy="workspace",
        )

    store.create_job.assert_not_awaited()


async def test_creation_rejects_a_removed_workspace_path(
    service: AgentJobService,
    store: Mock,
    manager: Mock,
    tmp_path: Path,
):
    """A durable identity cannot admit new work after its canonical directory disappears."""
    store.resolve_workspace.return_value = make_workspace(canonical_path=tmp_path / "removed")

    with pytest.raises(WorkspaceInvalidError):
        await service.start(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            backend_id="codex",
            operation=TurnOperation(prompt="Inspect"),
            explicit_config=ExecutionConfigValues(),
        )

    manager.require_operation.assert_not_called()
    store.create_job.assert_not_awaited()


async def test_review_rejects_unsupported_operation_without_creating_job(
    service: AgentJobService,
    store: Mock,
    manager: Mock,
):
    """An unadvertised operation fails on static capabilities before configuration or storage."""
    store.get_session.return_value = make_agent_session(backend_id="legacy-claude")
    manager.require_operation.side_effect = UnsupportedCapabilityError("legacy-claude", "review")

    with pytest.raises(UnsupportedCapabilityError):
        await service.review(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            session_id="session-test",
            operation=make_review_operation(),
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()
    manager.list_statuses.assert_not_awaited()


@pytest.mark.parametrize(
    ("capability_update", "operation"),
    [
        (
            {"review_deliveries": frozenset({"inline"})},
            make_review_operation(delivery="detached"),
        ),
        (
            {"review_targets": frozenset({"commit"})},
            make_review_operation(target=ReviewTarget(kind="working_tree")),
        ),
    ],
)
async def test_review_rejects_unadvertised_delivery_or_target(
    service: AgentJobService,
    store: Mock,
    backend: Mock,
    capability_update: dict[str, object],
    operation: ReviewOperation,
):
    """Review sub-capabilities cannot be inferred from the broad operation bit."""
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(update=capability_update)
        }
    )

    with pytest.raises(UnsupportedCapabilityError):
        await service.review(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            session_id="session-test",
            operation=operation,
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()


async def test_fork_requires_the_session_fork_capability(
    service: AgentJobService,
    store: Mock,
    backend: Mock,
):
    """Advertising the fork operation alone cannot authorize child-session semantics."""
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(
                update={"session_fork": False}
            )
        }
    )

    with pytest.raises(UnsupportedCapabilityError):
        await service.fork_session(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            session_id="session-test",
            operation=ForkOperation(prompt="Fork"),
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()


async def test_unadvertised_sandbox_is_rejected_before_snapshot_or_create(
    service: AgentJobService,
    store: Mock,
    backend: Mock,
    config_resolver: Mock,
):
    """A requested sandbox cannot exceed the backend's static advertised set."""
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(
                update={"sandbox_modes": frozenset({"read_only"})}
            )
        }
    )

    with pytest.raises(UnsupportedCapabilityError):
        await service.start(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            backend_id="codex",
            operation=TurnOperation(prompt="Inspect"),
            explicit_config=ExecutionConfigValues(sandbox="workspace_write"),
        )

    config_resolver.snapshot.assert_not_called()
    store.create_job.assert_not_awaited()


@pytest.mark.parametrize("sandbox", [None, "read_only"])
async def test_never_approval_requires_an_explicit_mutating_sandbox(
    service: AgentJobService,
    store: Mock,
    sandbox: str | None,
):
    """Approval suppression cannot be admitted for an implicit or read-only sandbox."""
    with pytest.raises(UnsupportedCapabilityError):
        await service.start(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            backend_id="codex",
            operation=TurnOperation(prompt="Inspect"),
            explicit_config=ExecutionConfigValues(
                sandbox=sandbox,
                approval_policy="never",
            ),
        )

    store.create_job.assert_not_awaited()


async def test_never_approval_accepts_advertised_explicit_mutating_sandbox(
    service: AgentJobService,
    store: Mock,
):
    """A static mutating sandbox makes the explicit never-approval combination admissible."""
    await service.start(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        backend_id="codex",
        operation=TurnOperation(prompt="Inspect"),
        explicit_config=ExecutionConfigValues(
            sandbox="workspace_write",
            approval_policy="never",
        ),
    )

    store.create_job.assert_awaited_once()


async def test_continue_uses_existing_session_policy_owner_and_checkpoint(
    service: AgentJobService,
    store: Mock,
):
    """Continuation must not replace the durable source session or broaden its identity."""
    store.get_session.return_value = make_agent_session(
        owner_id="local:400", access_policy="workspace"
    )
    operation = TurnOperation(prompt="Continue")

    handle = await service.continue_session(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(principal_id="local:501"),
        session_id="session-test",
        operation=operation,
        explicit_config=ExecutionConfigValues(),
        idempotency_key="continue-1",
    )

    command = store.create_job.await_args.args[0]
    assert handle.session_id == "session-test"
    assert command.session_id == "session-test"
    assert command.create_session is False
    assert command.parent_session_id is None
    assert command.source_session_id == "session-test"
    assert command.source_checkpoint == (ProviderReference(kind="thread", value="thread-test"),)
    assert command.owner_id == "local:400"
    assert command.access_policy == "workspace"
    assert command.command_family == "continue_session"
    assert command.idempotency_key == "continue-1"


async def test_private_session_creation_access_violation_is_access_denied(
    service: AgentJobService,
    store: Mock,
):
    """Creation access failures stay explicit instead of disclosing via read-style not-found."""
    store.get_session.return_value = make_agent_session(owner_id="local:400")

    with pytest.raises(AccessDeniedError):
        await service.continue_session(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(principal_id="local:501"),
            session_id="session-test",
            operation=TurnOperation(prompt="Continue"),
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()


async def test_source_session_workspace_mismatch_is_access_denied(
    service: AgentJobService,
    store: Mock,
):
    """A session cannot be continued through a different durable workspace selector."""
    store.get_session.return_value = make_agent_session(workspace_id="ws-other")

    with pytest.raises(AccessDeniedError):
        await service.continue_session(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            session_id="session-test",
            operation=TurnOperation(prompt="Continue"),
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()


async def test_fork_creates_child_from_parent_checkpoint(
    service: AgentJobService,
    store: Mock,
):
    """A fork gets a new session while idempotency and provider state derive from its parent."""
    operation = ForkOperation(prompt="Try another approach")

    handle = await service.fork_session(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        session_id="session-test",
        operation=operation,
        explicit_config=ExecutionConfigValues(),
    )

    command = store.create_job.await_args.args[0]
    assert handle.session_id == command.session_id
    assert command.session_id != "session-test"
    assert command.create_session is True
    assert command.parent_session_id == "session-test"
    assert command.source_session_id == "session-test"
    assert command.source_checkpoint == (ProviderReference(kind="thread", value="thread-test"),)
    assert command.command_family == "fork_session"


@pytest.mark.parametrize(
    ("delivery", "create_session", "parent_session_id"),
    [("inline", False, None), ("detached", True, "session-test")],
)
async def test_review_delivery_selects_existing_or_child_session_semantics(
    service: AgentJobService,
    store: Mock,
    delivery: str,
    create_session: bool,
    parent_session_id: str | None,
):
    """Inline review continues the source while detached review inherits into a child."""
    operation = make_review_operation(delivery=delivery)

    handle = await service.review(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        session_id="session-test",
        operation=operation,
        explicit_config=ExecutionConfigValues(),
    )

    command = store.create_job.await_args.args[0]
    assert command.create_session is create_session
    assert command.parent_session_id == parent_session_id
    assert command.source_session_id == "session-test"
    assert command.source_checkpoint == (ProviderReference(kind="thread", value="thread-test"),)
    if delivery == "inline":
        assert handle.session_id == "session-test"
    else:
        assert handle.session_id == command.session_id
        assert command.session_id != "session-test"


async def test_diagnose_is_sessionless_and_does_not_check_backend_health(
    service: AgentJobService,
    store: Mock,
    manager: Mock,
):
    """Diagnostics admission queues provider work without pre-running its health operation."""
    handle = await service.diagnose(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        backend_id="codex",
        explicit_config=ExecutionConfigValues(),
        idempotency_key="diagnose-1",
    )

    command = store.create_job.await_args.args[0]
    assert handle.session_id is None
    assert command.operation.kind == "diagnostics"
    assert command.session_id is None
    assert command.create_session is False
    assert command.command_family == "diagnose"
    manager.list_statuses.assert_not_awaited()


async def test_private_job_is_not_disclosed_to_other_principal(
    service: AgentJobService,
    store: Mock,
):
    """Private object reads map unauthorized access to the same stable not-found error."""
    store.get_job.return_value = make_agent_job(owner_id="local:501")

    with pytest.raises(JobNotFoundError):
        await service.status(
            workspace=WorkspaceSelector(workspace_id="ws-test"),
            access=make_access_context(principal_id="local:502"),
            job_id="job-test",
        )

    store.get_job_attempts.assert_not_awaited()


async def test_workspace_job_is_visible_to_authorized_principal(
    service: AgentJobService,
    store: Mock,
):
    """Workspace policy grants reads only through explicit trusted workspace authority."""
    store.get_job.return_value = make_agent_job(owner_id="local:400", access_policy="workspace")

    status = await service.status(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(principal_id="local:502"),
        job_id="job-test",
    )

    assert status.job_id == "job-test"


async def test_object_read_rejects_a_different_workspace_without_disclosure(
    service: AgentJobService,
    store: Mock,
):
    """A valid job id cannot escape the workspace selector's durable scope."""
    store.get_job.return_value = make_agent_job(workspace_id="ws-other")

    with pytest.raises(JobNotFoundError):
        await service.result(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            job_id="job-test",
        )


async def test_status_survives_removed_directory_and_projects_complete_snapshot(
    service: AgentJobService,
    store: Mock,
    tmp_path: Path,
):
    """Durable status remains available by workspace id and includes worker and event state."""
    removed = tmp_path / "removed"
    assert not removed.exists()
    store.resolve_workspace.return_value = make_workspace(canonical_path=removed)
    resolved_config = ResolvedExecutionConfig(model="gpt-5", sources={"model": "provider"})
    store.get_job.return_value = make_agent_job(
        state="input_required",
        resolved_config=resolved_config,
        cancel_requested_at=NOW,
    )
    store.get_job_attempts.return_value = (
        JobAttempt(job_id="job-test", attempt_number=1, phase="executing"),
    )
    pending = make_pending_permission()
    store.get_pending_inputs.return_value = (pending,)
    store.read_events.return_value = EventPage(
        events=(
            JobEvent(
                job_id="job-test",
                sequence=7,
                type="input_required",
                occurred_at=NOW,
            ),
        )
    )

    status = await service.status(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id="job-test",
    )

    assert status.phase == "executing"
    assert status.pending_inputs == (pending,)
    assert status.resolved_config == resolved_config
    assert status.latest_event_sequence == 7
    assert status.cancel_requested is True


async def test_status_reads_all_event_pages_for_latest_sequence(
    service: AgentJobService,
    store: Mock,
):
    """A status projection cannot report a stale cursor when event history exceeds one page."""
    store.read_events.side_effect = [
        EventPage(
            events=(JobEvent(job_id="job-test", sequence=1000, type="progress", occurred_at=NOW),),
            next_after_sequence=1000,
            has_more=True,
        ),
        EventPage(
            events=(JobEvent(job_id="job-test", sequence=1001, type="progress", occurred_at=NOW),)
        ),
    ]

    status = await service.status(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id="job-test",
    )

    assert status.latest_event_sequence == 1001
    assert [call.args[1] for call in store.read_events.await_args_list] == [0, 1000]


@pytest.mark.parametrize(
    ("state", "stored_result", "response_type", "status"),
    [
        ("queued", None, PendingJobResultResponse, "pending"),
        ("running", None, PendingJobResultResponse, "pending"),
        ("input_required", None, PendingJobResultResponse, "pending"),
        (
            "completed",
            JobResultEnvelope(job_id="job-test", payload=make_turn_result(), completed_at=NOW),
            SucceededJobResultResponse,
            "succeeded",
        ),
        ("failed", make_job_error(), FailedJobResultResponse, "failed"),
        ("cancelled", None, CancelledJobResultResponse, "cancelled"),
    ],
)
async def test_result_returns_strict_state_discriminated_union(
    service: AgentJobService,
    store: Mock,
    state: str,
    stored_result: object,
    response_type: type,
    status: str,
):
    """Each durable state maps to exactly one public result-poll variant."""
    store.get_job.return_value = make_agent_job(state=state)
    store.get_job_result.return_value = stored_result

    response = await service.result(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id="job-test",
    )

    assert isinstance(response, response_type)
    assert response.status == status


async def test_list_jobs_uses_store_access_filter_and_enriches_items(
    service: AgentJobService,
    store: Mock,
):
    """Listing delegates durable visibility while retaining the complete status projection."""
    job = make_agent_job(owner_id="local:400", access_policy="workspace")
    store.list_jobs.return_value = StoredJobPage(jobs=(job,), next_cursor="next")

    page = await service.list_jobs(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(principal_id="local:502"),
    )

    query = store.list_jobs.await_args.args[0]
    assert query.workspace_id == "ws-test"
    assert query.states == ALL_STATES
    assert query.access.principal_id == "local:502"
    assert query.access.workspace_authorized is True
    assert [item.job_id for item in page.items] == ["job-test"]
    assert page.next_cursor == "next"


async def test_queued_cancel_is_immediate_without_backend_capability(
    service: AgentJobService,
    store: Mock,
    backend: Mock,
    notifier: EventNotifier,
):
    """Queued work has no provider execution to interrupt and always cancels in storage."""
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(
                update={"cancellation": False}
            )
        }
    )
    store.get_job.return_value = make_agent_job(state="queued")

    receipt = await service.cancel(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id="job-test",
    )

    command = store.request_cancel.await_args.args[0]
    assert receipt.completed_immediately is True
    assert command.event.type == "job_cancelled"
    assert notifier.revision == 1


async def test_active_cancel_without_backend_capability_records_no_intent(
    service: AgentJobService,
    store: Mock,
    backend: Mock,
    notifier: EventNotifier,
):
    """An active provider that cannot cancel must not receive a false durable intent."""
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(
                update={"cancellation": False}
            )
        }
    )
    store.get_job.return_value = make_agent_job(state="running")

    with pytest.raises(UnsupportedCapabilityError):
        await service.cancel(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            job_id="job-test",
        )

    store.request_cancel.assert_not_awaited()
    assert notifier.revision == 0


async def test_active_cancel_records_intent_event_after_commit(
    service: AgentJobService,
    store: Mock,
    notifier: EventNotifier,
):
    """A supported active cancellation records semantic intent before waking workers."""

    async def request_cancel(command):
        assert notifier.revision == 0
        return CancelReceipt(
            job_id="job-test",
            state="running",
            cancel_requested=True,
            completed_immediately=False,
        )

    store.get_job.return_value = make_agent_job(state="running")
    store.request_cancel.side_effect = request_cancel

    receipt = await service.cancel(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id="job-test",
    )

    command = store.request_cancel.await_args.args[0]
    assert receipt.cancel_requested is True
    assert command.event.type == "cancel_requested"
    assert notifier.revision == 1


async def test_terminal_cancel_is_an_idempotent_no_op_without_wake_or_capability(
    service: AgentJobService,
    store: Mock,
    backend: Mock,
    notifier: EventNotifier,
):
    """Terminal cancellation polling delegates the stable receipt without a false new event."""
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(
                update={"cancellation": False}
            )
        }
    )
    store.get_job.return_value = make_agent_job(state="completed")
    store.request_cancel.return_value = CancelReceipt(
        job_id="job-test",
        state="completed",
        cancel_requested=False,
        completed_immediately=False,
    )

    receipt = await service.cancel(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id="job-test",
    )

    assert receipt.state == "completed"
    assert notifier.revision == 0


async def test_input_response_replay_and_conflict_preserve_single_wake(
    service: AgentJobService,
    store: Mock,
    notifier: EventNotifier,
):
    """Only the first committed response emits an event; replay and conflict remain idempotent."""
    store.get_job.return_value = make_agent_job(state="input_required")
    store.resolve_input.side_effect = [
        InputResolutionReceipt(job_id="job-test", input_id="input-test"),
        InputResolutionReceipt(job_id="job-test", input_id="input-test", replayed=True),
        InputAlreadyResolvedError("job-test", "input-test"),
    ]
    response = PermissionResponse(granted=frozenset())

    first = await service.respond(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id="job-test",
        input_id="input-test",
        response=response,
    )
    replay = await service.respond(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id="job-test",
        input_id="input-test",
        response=response,
    )
    with pytest.raises(InputAlreadyResolvedError):
        await service.respond(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            job_id="job-test",
            input_id="input-test",
            response=PermissionResponse(granted=frozenset({"different"})),
        )

    commands = [call.args[0] for call in store.resolve_input.await_args_list]
    assert first.replayed is False
    assert replay.replayed is True
    assert all(command.event.type == "input_resolved" for command in commands)
    assert notifier.revision == 1


async def test_subscription_authorizes_before_first_event_read(
    service: AgentJobService,
    store: Mock,
):
    """The synchronous subscription factory defers async access without exposing history."""
    store.get_job.return_value = make_agent_job(owner_id="local:501")

    subscription = service.subscribe_events(
        workspace=WORKSPACE_SELECTOR,
        access=make_access_context(principal_id="local:502"),
        job_id="job-test",
    )

    assert isinstance(subscription, JobEventSubscription)
    store.get_job.assert_not_awaited()
    with pytest.raises(JobNotFoundError):
        await anext(subscription)
    store.read_events.assert_not_awaited()


async def test_list_backends_requires_workspace_authority_and_runs_health_only_there(
    service: AgentJobService,
    manager: Mock,
    backend: Mock,
):
    """Backend health is an explicit discovery query, never an admission side effect."""
    status = BackendStatus(
        descriptor=backend.descriptor,
        availability=BackendAvailability(available=True),
    )
    manager.list_statuses.return_value = (status,)

    with pytest.raises(AccessDeniedError):
        await service.list_backends(
            workspace=WORKSPACE_SELECTOR,
            access=make_access_context(),
        )

    statuses = await service.list_backends(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
    )

    assert statuses == (status,)
    manager.list_statuses.assert_awaited_once()
