"""Canonical application-service admission, access, query, and control behavior."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from nexus_mcp.core import (
    BackendEvent,
    CancelReceipt,
    ExecutionConfigValues,
    InputAlreadyResolvedError,
    InputResolutionReceipt,
    PermissionResponse,
    TurnOperation,
    UnsupportedCapabilityError,
)
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.service import AgentJobService
from nexus_mcp.jobs.store import SucceededTerminalOutcome
from tests.fixtures import (
    make_agent_job,
    make_pending_permission,
    make_turn_result,
)
from tests.unit.jobs._service_support import (
    NOW,
    WORKSPACE_SELECTOR,
    authorized_access,
)


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
    assert command.active_cancellation_allowed is False
    assert command.queued_event.type == "job_cancelled"
    assert command.active_event.type == "cancel_requested"
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
    store.request_cancel.return_value = CancelReceipt(
        job_id="job-test",
        state="running",
        cancel_requested=False,
        completed_immediately=False,
        event_committed=False,
    )

    with pytest.raises(UnsupportedCapabilityError):
        await service.cancel(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            job_id="job-test",
        )

    command = store.request_cancel.await_args.args[0]
    assert command.active_cancellation_allowed is False
    assert notifier.revision == 0


async def test_queued_to_running_cancel_race_raises_without_false_wake(
    service: AgentJobService,
    store: Mock,
    backend: Mock,
    notifier: EventNotifier,
):
    """The atomic store result overrides the stale queued snapshot used for authorization."""
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(
                update={"cancellation": False}
            )
        }
    )
    store.get_job.return_value = make_agent_job(state="queued")
    store.request_cancel.return_value = CancelReceipt(
        job_id="job-test",
        state="running",
        cancel_requested=False,
        completed_immediately=False,
        event_committed=False,
    )

    with pytest.raises(UnsupportedCapabilityError):
        await service.cancel(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            job_id="job-test",
        )

    assert notifier.revision == 0


async def test_concurrent_service_cancel_notifies_only_the_event_winner(
    service: AgentJobService,
    store: Mock,
    notifier: EventNotifier,
):
    """Receipt event truth prevents a concurrent cancellation replay from emitting a wake."""
    store.get_job.return_value = make_agent_job(state="running")
    store.request_cancel.side_effect = [
        CancelReceipt(
            job_id="job-test",
            state="running",
            cancel_requested=True,
            completed_immediately=False,
            event_committed=True,
        ),
        CancelReceipt(
            job_id="job-test",
            state="running",
            cancel_requested=True,
            completed_immediately=False,
            event_committed=False,
        ),
    ]

    receipts = await asyncio.gather(
        service.cancel(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            job_id="job-test",
        ),
        service.cancel(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            job_id="job-test",
        ),
    )

    assert sum(receipt.event_committed for receipt in receipts) == 1
    assert notifier.revision == 1


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
            event_committed=True,
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
    assert command.active_cancellation_allowed is True
    assert command.queued_event.type == "job_cancelled"
    assert command.active_event.type == "cancel_requested"
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
        event_committed=False,
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


@pytest.mark.parametrize("terminal", [False, True], ids=["running", "terminal"])
async def test_real_service_replays_resolved_input_after_job_state_changes(
    real_service_environment,
    terminal: bool,
):
    """Replay and conflict semantics survive running or terminal job transitions without wakes."""
    service, durable_store, _, notifier = real_service_environment
    handle = await service.start(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        backend_id="codex",
        operation=TurnOperation(prompt="Request input"),
        explicit_config=ExecutionConfigValues(),
    )
    claimed = await durable_store.claim_next(
        "worker-input-service",
        datetime(2099, 1, 1, tzinfo=UTC),
        event=BackendEvent(type="progress", occurred_at=NOW),
    )
    assert claimed is not None and claimed.job.job_id == handle.job_id
    await durable_store.mark_running(
        claimed.token,
        (),
        event=BackendEvent(type="job_started", occurred_at=NOW),
    )
    pending = make_pending_permission(job_id=handle.job_id, created_at=NOW)
    await durable_store.mark_input_required(
        claimed.token,
        (pending,),
        event=BackendEvent(type="input_required", occurred_at=NOW),
    )
    response = PermissionResponse(granted=frozenset({"network:api.example.com"}))
    await service.respond(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id=handle.job_id,
        input_id=pending.input_id,
        response=response,
    )
    await durable_store.mark_running(
        claimed.token,
        (pending.input_id,),
        event=BackendEvent(type="job_started", occurred_at=NOW),
    )
    if terminal:
        await durable_store.terminalize(
            claimed.token,
            SucceededTerminalOutcome(result=make_turn_result(), completed_at=NOW),
            event=BackendEvent(type="job_completed", occurred_at=NOW),
        )
    revision_before_replay = notifier.revision

    replay = await service.respond(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id=handle.job_id,
        input_id=pending.input_id,
        response=response,
    )
    with pytest.raises(InputAlreadyResolvedError):
        await service.respond(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            job_id=handle.job_id,
            input_id=pending.input_id,
            response=PermissionResponse(granted=frozenset()),
        )

    events = await durable_store.read_events(handle.job_id, 0, 20)
    assert replay.replayed is True
    assert notifier.revision == revision_before_replay
    assert [event.type for event in events.events].count("input_resolved") == 1
