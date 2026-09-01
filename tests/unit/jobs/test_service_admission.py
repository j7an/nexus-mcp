"""Canonical application-service admission, access, query, and control behavior."""

from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from nexus_mcp.backends import BackendManager
from nexus_mcp.core import (
    AccessDeniedError,
    BackendEvent,
    ExecutionConfigValues,
    ForkOperation,
    RequestedExecutionConfig,
    ReviewOperation,
    ReviewTarget,
    SessionBusyError,
    TurnOperation,
    UnsupportedCapabilityError,
    WorkspaceInvalidError,
    WorkspaceSelector,
)
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.service import AgentJobService
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore
from nexus_mcp.jobs.store import (
    CreateJobCommand,
    CreateJobResult,
)
from tests.fixtures import (
    make_access_context,
    make_agent_session,
    make_job_handle,
    make_workspace,
)
from tests.job_fakes import ScriptedBackend
from tests.unit.jobs._service_support import (
    NOW,
    WORKSPACE_SELECTOR,
    _source_session,
    _StableCaptureResolver,
    authorized_access,
    make_review_operation,
)

if TYPE_CHECKING:
    from nexus_mcp.jobs.configuration import NexusConfigResolver


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


async def test_service_starts_with_fresh_sqlite_workspace_without_mcp_policy(
    tmp_path: Path,
):
    """The framework-independent service owns first-use workspace admission."""
    workspace_path = tmp_path / "fresh-workspace"
    workspace_path.mkdir()
    durable_store = SQLiteJobStore(tmp_path / "fresh-service.sqlite3")
    await durable_store.open()
    backend = ScriptedBackend(backend_id="codex")
    service = AgentJobService(
        store=durable_store,
        backend_manager=BackendManager([backend]),
        config_resolver=cast("NexusConfigResolver", _StableCaptureResolver()),
        notifier=EventNotifier(),
    )
    try:
        handle = await service.start(
            workspace=WorkspaceSelector(path=workspace_path),
            access=make_access_context(authorize_local_workspaces=True),
            backend_id="codex",
            operation=TurnOperation(prompt="Start without MCP"),
            explicit_config=ExecutionConfigValues(),
        )
        workspace = await durable_store.resolve_workspace(WorkspaceSelector(path=workspace_path))
    finally:
        await durable_store.close()

    assert handle.session_id is not None
    assert workspace.canonical_path == workspace_path.resolve()


@pytest.mark.parametrize("family", ["fork", "review_detached"])
async def test_child_service_admission_loses_source_idle_race_atomically(
    real_service_environment,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
):
    """The store, not a stale service read, decides whether a child source is idle."""
    service, durable_store, _, _ = real_service_environment
    source_session_id = await _source_session(service)
    original_create = durable_store.create_job
    raced = False

    async def create_after_source_becomes_busy(command: CreateJobCommand):
        nonlocal raced
        if command.parent_session_id == source_session_id and not raced:
            raced = True
            await original_create(
                CreateJobCommand(
                    workspace=make_workspace(canonical_path=command.workspace.canonical_path),
                    backend_id=command.backend_id,
                    owner_id=command.owner_id,
                    access_policy=command.access_policy,
                    operation=TurnOperation(prompt="Racing source turn"),
                    requested_config=RequestedExecutionConfig(),
                    session_id=source_session_id,
                    create_session=False,
                    command_family="race",
                    queued_event=BackendEvent(type="job_queued", occurred_at=NOW),
                )
            )
        return await original_create(command)

    monkeypatch.setattr(durable_store, "create_job", create_after_source_becomes_busy)
    with pytest.raises(SessionBusyError) as raised:
        if family == "fork":
            await service.fork_session(
                workspace=WORKSPACE_SELECTOR,
                access=authorized_access(),
                session_id=source_session_id,
                operation=ForkOperation(prompt="Fork after stale read"),
                explicit_config=ExecutionConfigValues(),
            )
        else:
            await service.review(
                workspace=WORKSPACE_SELECTOR,
                access=authorized_access(),
                session_id=source_session_id,
                operation=make_review_operation(delivery="detached"),
                explicit_config=ExecutionConfigValues(),
            )

    assert raced is True
    assert raised.value.session_id == source_session_id


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
    store.resolve_or_create_workspace.return_value = make_workspace(
        canonical_path=tmp_path / "removed"
    )

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


@pytest.mark.parametrize("path_kind", ["missing", "file"])
async def test_unauthorized_creation_does_not_disclose_workspace_liveness(
    service: AgentJobService,
    store: Mock,
    manager: Mock,
    tmp_path: Path,
    path_kind: str,
):
    """Authorization must reject before missing versus non-directory path probes diverge."""
    canonical_path = tmp_path / path_kind
    if path_kind == "file":
        canonical_path.write_text("not a directory", encoding="utf-8")
    store.resolve_workspace.return_value = make_workspace(canonical_path=canonical_path)

    with pytest.raises(AccessDeniedError):
        await service.start(
            workspace=WORKSPACE_SELECTOR,
            access=make_access_context(),
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
