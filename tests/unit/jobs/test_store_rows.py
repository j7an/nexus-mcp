"""Focused contracts for pure durable-store row hydration."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus_mcp.core import (
    PermissionRequest,
    ProviderReference,
    RequestedExecutionConfig,
    TurnOperation,
    TurnResult,
)
from nexus_mcp.jobs._store_rows import (
    _event_from_row,
    _job_attempt_from_row,
    _job_from_row,
    _job_result_from_row,
    _pending_input_from_row,
    _provider_references_from_pairs,
    _session_from_row,
    _workspace_from_row,
)

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def test_workspace_row_hydration_constructs_the_durable_identity() -> None:
    """Workspace columns must hydrate without consulting a database connection."""
    canonical_path = Path.cwd().resolve()
    workspace = _workspace_from_row(
        {
            "workspace_id": "ws-test",
            "canonical_path": str(canonical_path),
            "display_name": "Test workspace",
            "config_ref": "nexus.toml",
            "created_at_ms": 0,
            "updated_at_ms": 1_000,
        }
    )

    assert workspace.workspace_id == "ws-test"
    assert workspace.canonical_path == canonical_path
    assert workspace.display_name == "Test workspace"
    assert workspace.config_reference == "nexus.toml"
    assert workspace.created_at == EPOCH
    assert workspace.updated_at == datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)


def test_session_row_hydration_uses_explicitly_fetched_provider_references() -> None:
    """Session construction must consume related references supplied by the store."""
    references = (ProviderReference(kind="thread", value="thread-1"),)
    session = _session_from_row(
        {
            "session_id": "session-1",
            "workspace_id": "ws-test",
            "backend_id": "codex",
            "owner_id": "local:501",
            "access_policy": "private",
            "parent_session_id": None,
            "created_at_ms": 0,
            "updated_at_ms": 0,
        },
        references,
    )

    assert session.session_id == "session-1"
    assert session.provider_references == references
    assert session.created_at == EPOCH


def test_job_row_hydration_uses_explicitly_fetched_source_checkpoint() -> None:
    """Job construction must not perform its former nested reference query."""
    operation = TurnOperation(prompt="Inspect the workspace")
    requested = RequestedExecutionConfig()
    checkpoint = (ProviderReference(kind="thread", value="thread-source"),)
    row: dict[str, Any] = {
        "job_id": "job-1",
        "workspace_id": "ws-test",
        "backend_id": "codex",
        "owner_id": "local:501",
        "operation_json": operation.model_dump_json(),
        "operation_schema_version": 1,
        "operation_kind": "turn",
        "requested_config_json": requested.model_dump_json(),
        "requested_config_schema_version": 1,
        "request_hash": "a" * 64,
        "access_policy": "private",
        "session_id": "session-1",
        "idempotency_key": None,
        "state": "queued",
        "resolved_config_json": None,
        "resolved_config_schema_version": None,
        "cancel_requested_at_ms": None,
        "lease_owner": None,
        "lease_generation": 0,
        "lease_expires_at_ms": None,
        "retry_at_ms": None,
        "created_at_ms": 0,
        "updated_at_ms": 0,
        "terminal_at_ms": None,
    }

    job = _job_from_row(row, checkpoint)

    assert job.job_id == "job-1"
    assert job.operation == operation
    assert job.requested_config == requested
    assert job.source_checkpoint == checkpoint
    assert job.lease_generation is None


def test_job_result_row_hydration_constructs_a_success_envelope() -> None:
    """A valid terminal result row must hydrate independently of its query."""
    result = TurnResult(message="Completed")
    hydrated = _job_result_from_row(
        "job-1",
        {
            "job_state": "completed",
            "result_job_id": "job-1",
            "outcome_kind": "succeeded",
            "payload_json": result.model_dump_json(),
            "payload_schema_version": 1,
            "error_json": None,
            "error_schema_version": None,
            "created_at_ms": 0,
        },
    )

    assert hydrated is not None
    assert hydrated.job_id == "job-1"
    assert hydrated.payload == result
    assert hydrated.completed_at == EPOCH


def test_attempt_row_hydration_preserves_error_summary_and_timestamps() -> None:
    """Attempt columns must hydrate without retaining the serialized error model."""
    attempt = _job_attempt_from_row(
        {
            "job_id": "job-1",
            "attempt_number": 2,
            "phase": "reconciling",
            "owner_id": "worker-1",
            "lease_generation": 3,
            "lease_expires_at_ms": 1_000,
            "heartbeat_at_ms": 0,
            "retry_classification": None,
            "reconciliation_classification": "provider_lookup",
            "started_at_ms": 0,
            "ended_at_ms": 1_000,
            "error_json": (
                '{"code":"outcome_unknown","message":"Provider outcome is unknown",'
                '"retry_disposition":"reconcile_required","recoverable":true,"details":{}}'
            ),
            "error_schema_version": 1,
        }
    )

    assert attempt.attempt_number == 2
    assert attempt.phase == "reconciling"
    assert attempt.lease_expires_at == datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)
    assert attempt.error_code == "outcome_unknown"
    assert attempt.error_message == "Provider outcome is unknown"


def test_pending_input_row_hydration_uses_explicit_provider_reference() -> None:
    """Pending input construction must receive its scoped provider reference explicitly."""
    request = PermissionRequest(prompt="Allow write?", requested=frozenset({"workspace_write"}))
    reference = ProviderReference(kind="request", value="request-1")
    pending = _pending_input_from_row(
        {
            "input_id": "input-1",
            "job_id": "job-1",
            "kind": "permission",
            "request_json": request.model_dump_json(),
            "request_schema_version": 1,
            "status": "pending",
            "response_json": None,
            "response_schema_version": None,
            "created_at_ms": 0,
            "resolved_at_ms": None,
        },
        reference,
    )

    assert pending.request == request
    assert pending.provider_reference == reference
    assert pending.response is None
    assert pending.resolved_at is None


def test_event_row_hydration_uses_explicit_provider_reference() -> None:
    """Event construction must receive its scoped provider reference explicitly."""
    reference = ProviderReference(kind="message", value="message-1")
    event = _event_from_row(
        {
            "job_id": "job-1",
            "sequence": 4,
            "event_type": "progress",
            "payload_json": '{"percent":50}',
            "payload_schema_version": 1,
            "created_at_ms": 0,
            "attempt_number": 2,
            "provider_event_type": "turn/progress",
        },
        reference,
    )

    assert event.sequence == 4
    assert dict(event.payload) == {"percent": 50}
    assert event.provider_reference == reference


def test_provider_reference_hydration_deduplicates_without_reordering() -> None:
    """Repeated provider-reference rows must retain first-seen ordering."""
    references = _provider_references_from_pairs(
        (("thread", "one"), ("message", "two"), ("thread", "one"))
    )

    assert references == (
        ProviderReference(kind="thread", value="one"),
        ProviderReference(kind="message", value="two"),
    )
