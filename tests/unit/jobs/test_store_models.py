"""Shared behavioral contract for every durable job-store implementation."""

import pytest
from pydantic import ValidationError

from nexus_mcp.core import (
    DiagnosticsOperation,
    JobEventType,
    ProviderReference,
)
from nexus_mcp.jobs.store import (
    CancelJobCommand,
    JobAccessFilter,
    JobQuery,
    PrunePolicy,
)
from tests.unit.jobs._store_contract_support import (
    NOW,
    make_create_job_command,
    make_event,
)


def test_create_command_requires_session_for_non_diagnostics():
    """A non-diagnostic operation cannot enter storage without a session identity."""
    with pytest.raises(ValidationError):
        make_create_job_command(session_id=None, create_session=False)


def test_create_command_keeps_diagnostics_sessionless():
    """Diagnostics cannot accidentally acquire conversation semantics."""
    with pytest.raises(ValidationError):
        make_create_job_command(operation=DiagnosticsOperation())

    command = make_create_job_command(
        operation=DiagnosticsOperation(),
        session_id=None,
        create_session=False,
    )
    assert command.session_id is None


def test_create_session_requires_session_id():
    """Session creation cannot commit an identity-free session record."""
    with pytest.raises(ValidationError):
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=True,
        )


def test_cancel_command_carries_atomic_queued_and_active_decisions():
    """The store must choose capability and semantic event from one locked job state."""
    command = CancelJobCommand(
        job_id="job-test",
        requested_at=NOW,
        active_cancellation_allowed=False,
        queued_event=make_event("job_cancelled"),
        active_event=make_event("cancel_requested"),
    )

    assert command.active_cancellation_allowed is False
    assert command.queued_event.type == "job_cancelled"
    assert command.active_event.type == "cancel_requested"


@pytest.mark.parametrize(
    ("queued_type", "active_type"),
    [("cancel_requested", "cancel_requested"), ("job_cancelled", "job_cancelled")],
)
def test_cancel_command_rejects_untruthful_state_event_types(
    queued_type: JobEventType,
    active_type: JobEventType,
):
    """Swapping state-specific event types would make durable cancellation history false."""
    with pytest.raises(ValidationError):
        CancelJobCommand(
            job_id="job-test",
            requested_at=NOW,
            active_cancellation_allowed=True,
            queued_event=make_event(queued_type),
            active_event=make_event(active_type),
        )


def test_source_checkpoint_requires_a_derived_source_session():
    """Provider checkpoints cannot be admitted without a session that owns them."""
    with pytest.raises(ValidationError):
        make_create_job_command(
            operation=DiagnosticsOperation(),
            session_id=None,
            create_session=False,
            source_checkpoint=(ProviderReference(kind="thread", value="thread-1"),),
        )


def test_create_command_exposes_derived_source_session():
    """Idempotency and checkpoint provenance use one explicit source-session derivation."""
    root = make_create_job_command(session_id="root", create_session=True)
    continuation = make_create_job_command(session_id="existing", create_session=False)
    child = make_create_job_command(
        session_id="child",
        create_session=True,
        parent_session_id="parent",
    )

    assert root.source_session_id is None
    assert continuation.source_session_id == "existing"
    assert child.source_session_id == "parent"


def test_job_query_requires_nonempty_states_and_bounded_limit():
    """List calls cannot become unbounded scans or meaningless state filters."""
    access = JobAccessFilter(principal_id="local:501")
    with pytest.raises(ValidationError):
        JobQuery(workspace_id="ws-test", access=access, states=frozenset(), limit=10)
    with pytest.raises(ValidationError):
        JobQuery(workspace_id="ws-test", access=access, states={"queued"}, limit=101)


def test_prune_policy_couples_raw_cutoff_and_positive_byte_cap():
    """Raw diagnostics cannot be pruned by age without an explicit byte ceiling."""
    with pytest.raises(ValidationError):
        PrunePolicy(raw_diagnostic_before=NOW)
    with pytest.raises(ValidationError):
        PrunePolicy(raw_diagnostic_max_bytes=1024)
    with pytest.raises(ValidationError):
        PrunePolicy(raw_diagnostic_before=NOW, raw_diagnostic_max_bytes=0)
