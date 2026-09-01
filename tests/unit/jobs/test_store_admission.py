"""Shared behavioral contract for every durable job-store implementation."""

import asyncio
from pathlib import Path

import pytest

from nexus_mcp.core import (
    IdempotencyConflictError,
    ProviderReference,
    SessionBusyError,
    SessionNotFoundError,
    TurnOperation,
    Workspace,
    WorkspaceInvalidError,
    WorkspaceSelector,
)
from tests.fixtures import make_pending_permission
from tests.unit.jobs._store_contract_support import (
    LEASE_UNTIL,
    NOW,
    WORKSPACE_PATH,
    make_cancel_job_command,
    make_create_job_command,
    make_event,
)


async def test_create_job_replays_matching_idempotency_key(admission_store):
    """The same scoped request key returns its original handle without another event."""
    command = make_create_job_command(idempotency_key="request-1")
    first = await admission_store.create_job(command)
    second = await admission_store.create_job(command)

    assert second.handle == first.handle
    assert second.created is False
    page = await admission_store.read_events(first.handle.job_id, after_sequence=0, limit=10)
    assert [event.sequence for event in page.events] == [1]


async def test_resolve_or_create_workspace_is_canonical_and_idempotent(
    admission_store,
    tmp_path: Path,
):
    """Core stores assign one durable identity to canonical path aliases."""
    workspace_path = tmp_path / "canonical-workspace"
    workspace_path.mkdir()
    alias_path = tmp_path / "workspace-alias"
    alias_path.symlink_to(workspace_path, target_is_directory=True)

    direct, alias = await asyncio.gather(
        admission_store.resolve_or_create_workspace(WorkspaceSelector(path=workspace_path)),
        admission_store.resolve_or_create_workspace(WorkspaceSelector(path=alias_path)),
    )

    assert direct == alias
    assert direct.canonical_path == workspace_path.resolve()
    assert (
        await admission_store.resolve_workspace(WorkspaceSelector(workspace_id=direct.workspace_id))
        == direct
    )


async def test_idempotency_key_rejects_a_different_request(admission_store):
    """A key cannot alias two different immutable operation requests."""
    await admission_store.create_job(make_create_job_command(idempotency_key="request-1"))

    with pytest.raises(IdempotencyConflictError):
        await admission_store.create_job(
            make_create_job_command(
                idempotency_key="request-1",
                operation=TurnOperation(prompt="Do something else"),
            )
        )


async def test_idempotency_replay_after_terminal_returns_original_handle(job_store):
    """A late retry cannot reopen a terminal job or change its original admission handle."""
    command = make_create_job_command(idempotency_key="request-terminal")
    first = await job_store.create_job(command)
    await job_store.request_cancel(make_cancel_job_command(first.handle.job_id))

    replay = await job_store.create_job(command)

    assert replay.created is False
    assert replay.handle == first.handle
    stored = await job_store.get_job(first.handle.job_id)
    assert stored is not None and stored.state == "cancelled"


async def test_same_idempotency_key_is_independent_across_source_sessions(admission_store):
    """Continuation idempotency is scoped to the session from which work derives."""
    for session_id in ("source-a", "source-b"):
        source = await admission_store.create_job(
            make_create_job_command(session_id=session_id, idempotency_key=None)
        )
        await admission_store.request_cancel(make_cancel_job_command(source.handle.job_id))

    first = await admission_store.create_job(
        make_create_job_command(
            session_id="child-a",
            parent_session_id="source-a",
            idempotency_key="same-key",
        )
    )
    second = await admission_store.create_job(
        make_create_job_command(
            session_id="child-b",
            parent_session_id="source-b",
            idempotency_key="same-key",
        )
    )

    assert first.created is True
    assert second.created is True
    assert second.handle.job_id != first.handle.job_id


async def test_second_nonterminal_job_for_session_is_rejected(admission_store):
    """One session cannot admit overlapping provider operations."""
    await admission_store.create_job(make_create_job_command(session_id="session-1"))

    with pytest.raises(SessionBusyError):
        await admission_store.create_job(make_create_job_command(session_id="session-1"))


@pytest.mark.parametrize("source_state", ["queued", "running", "input_required"])
async def test_child_admission_rejects_busy_source_session_atomically(
    job_store,
    source_state: str,
):
    """Fork-like child admission cannot snapshot a source with active provider work."""
    source = await job_store.create_job(
        make_create_job_command(session_id="source-session", idempotency_key=None)
    )
    if source_state != "queued":
        claimed = await job_store.claim_next(
            "source-worker",
            LEASE_UNTIL,
            event=make_event("job_started"),
        )
        assert claimed is not None
        await job_store.mark_running(
            claimed.token,
            (),
            event=make_event("job_started"),
        )
        if source_state == "input_required":
            await job_store.mark_input_required(
                claimed.token,
                (make_pending_permission(job_id=source.handle.job_id),),
                event=make_event("input_required"),
            )

    child = make_create_job_command(
        session_id="child-session",
        parent_session_id="source-session",
        idempotency_key="child-request",
    )
    with pytest.raises(SessionBusyError) as raised:
        await job_store.create_job(child)

    assert raised.value.session_id == "source-session"
    assert await job_store.get_session("child-session") is None


async def test_terminal_session_accepts_a_new_job(job_store):
    """The per-session uniqueness fence releases only after terminal cancellation."""
    first = await job_store.create_job(make_create_job_command(session_id="session-1"))
    receipt = await job_store.request_cancel(make_cancel_job_command(first.handle.job_id))

    second = await job_store.create_job(make_create_job_command(session_id="session-1"))

    assert receipt.completed_immediately is True
    assert second.handle.job_id != first.handle.job_id


async def test_claimed_queued_cancellation_closes_the_attempt(job_store):
    """Immediate queued cancellation cannot leave its claiming attempt active."""
    created = await job_store.create_job(make_create_job_command())
    claimed = await job_store.claim_next(
        "worker-1", LEASE_UNTIL, event=make_event("progress", action="claimed")
    )
    assert claimed is not None

    await job_store.request_cancel(make_cancel_job_command(created.handle.job_id))

    attempts = await job_store.get_job_attempts(created.handle.job_id)
    assert attempts[-1].phase == "finalizing"
    assert attempts[-1].ended_at == NOW


async def test_create_persists_workspace_session_and_queued_event(admission_store):
    """Admission atomically persists its durable identities and initial history."""
    command = make_create_job_command()
    created = await admission_store.create_job(command)

    assert (
        await admission_store.resolve_workspace(WorkspaceSelector(workspace_id="ws-test"))
        == command.workspace
    )
    assert (
        await admission_store.resolve_workspace(WorkspaceSelector(path=WORKSPACE_PATH))
        == command.workspace
    )
    session = await admission_store.get_session("session-test")
    assert session is not None
    assert session.workspace_id == "ws-test"
    events = await admission_store.read_events(created.handle.job_id, after_sequence=0, limit=10)
    assert [(event.sequence, event.type) for event in events.events] == [(1, "job_queued")]


async def test_create_round_trips_typed_job_and_session_snapshots(admission_store):
    """Admission reads reconstruct closed job and session models without untyped payloads."""
    command = make_create_job_command(idempotency_key="round-trip")
    created = await admission_store.create_job(command)

    job = await admission_store.get_job(created.handle.job_id)
    session = await admission_store.get_session("session-test")

    assert job is not None
    assert job.operation == command.operation
    assert job.requested_config == command.requested_config
    assert job.owner_id == command.owner_id
    assert job.access_policy == command.access_policy
    assert job.idempotency_key == command.idempotency_key
    assert job.source_checkpoint == command.source_checkpoint
    assert session is not None
    assert session.workspace_id == command.workspace.workspace_id
    assert session.backend_id == command.backend_id
    assert session.owner_id == command.owner_id
    assert session.access_policy == command.access_policy


async def test_failed_create_job_leaves_no_partial_workspace(admission_store, tmp_path: Path):
    """An admission failure rolls back every identity that would have accompanied the job."""
    command = make_create_job_command(
        workspace=Workspace(
            workspace_id="ws-rollback",
            canonical_path=tmp_path / "nexus-rollback",
            created_at=NOW,
            updated_at=NOW,
        ),
        session_id="missing-session",
        create_session=False,
    )

    with pytest.raises(SessionNotFoundError):
        await admission_store.create_job(command)
    with pytest.raises(WorkspaceInvalidError):
        await admission_store.resolve_workspace(WorkspaceSelector(workspace_id="ws-rollback"))


async def test_source_checkpoint_requires_existing_source_before_any_mutation(
    admission_store,
    tmp_path: Path,
):
    """A missing checkpoint source rolls back workspace, child session, job, event, and key."""
    workspace = Workspace(
        workspace_id="ws-missing-source",
        canonical_path=tmp_path / "nexus-missing-source",
        created_at=NOW,
        updated_at=NOW,
    )
    command = make_create_job_command(
        workspace=workspace,
        session_id="child-missing-source",
        parent_session_id="missing-source",
        source_checkpoint=(ProviderReference(kind="thread", value="thread-missing"),),
        idempotency_key="missing-source-key",
    )

    with pytest.raises(SessionNotFoundError):
        await admission_store.create_job(command)
    assert await admission_store.get_session("child-missing-source") is None
    with pytest.raises(WorkspaceInvalidError):
        await admission_store.resolve_workspace(
            WorkspaceSelector(workspace_id=workspace.workspace_id)
        )
