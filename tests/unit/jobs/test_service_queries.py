"""Canonical application-service admission, access, query, and control behavior."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from nexus_mcp.core import (
    AccessDeniedError,
    BackendAvailability,
    BackendStatus,
    CancelledJobResultResponse,
    FailedJobResultResponse,
    JobAttempt,
    JobEvent,
    JobNotFoundError,
    JobResultEnvelope,
    PendingJobResultResponse,
    ResolvedExecutionConfig,
    SucceededJobResultResponse,
    WorkspaceSelector,
)
from nexus_mcp.jobs.events import JobEventSubscription
from nexus_mcp.jobs.service import AgentJobService
from nexus_mcp.jobs.store import (
    EventPage,
    StoredJobPage,
)
from tests.fixtures import (
    make_access_context,
    make_agent_job,
    make_job_error,
    make_pending_permission,
    make_turn_result,
    make_workspace,
)
from tests.unit.jobs._service_support import (
    NOW,
    WORKSPACE_SELECTOR,
    authorized_access,
)

ALL_STATES = frozenset({"queued", "running", "input_required", "completed", "failed", "cancelled"})


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
