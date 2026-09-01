"""Shared non-fixture helpers for durable job-store contract tests."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from nexus_mcp.core import (
    BackendEvent,
    JobEventType,
    RequestedExecutionConfig,
    TurnOperation,
    Workspace,
)
from nexus_mcp.jobs.store import CancelJobCommand, CreateJobCommand

NOW = datetime(2026, 8, 30, 20, 0, tzinfo=UTC)
OLD = datetime(2025, 1, 1, tzinfo=UTC)
LEASE_UNTIL = datetime(2099, 1, 1, tzinfo=UTC)
WORKSPACE_PATH = Path(__file__).resolve().parents[3]


def make_event(event_type: JobEventType = "progress", **payload: JsonValue) -> BackendEvent:
    """Build one stable normalized event for store assertions."""
    return BackendEvent(type=event_type, payload=payload, occurred_at=NOW)


def make_create_job_command(**overrides: Any) -> CreateJobCommand:
    """Build a stable admitted turn request with a new durable session."""
    defaults: dict[str, object] = {
        "workspace": Workspace(
            workspace_id="ws-test",
            canonical_path=WORKSPACE_PATH,
            created_at=NOW,
            updated_at=NOW,
        ),
        "backend_id": "codex",
        "owner_id": "local:501",
        "access_policy": "private",
        "operation": TurnOperation(prompt="Inspect the workspace"),
        "requested_config": RequestedExecutionConfig(),
        "session_id": "session-test",
        "create_session": True,
        "command_family": "submit",
        "queued_event": make_event("job_queued", status="queued"),
    }
    return CreateJobCommand(**(defaults | overrides))


def make_cancel_job_command(
    job_id: str,
    *,
    active_cancellation_allowed: bool = True,
) -> CancelJobCommand:
    """Build one atomic cancellation decision with truthful state-specific events."""
    return CancelJobCommand(
        job_id=job_id,
        requested_at=NOW,
        active_cancellation_allowed=active_cancellation_allowed,
        queued_event=make_event("job_cancelled"),
        active_event=make_event("cancel_requested"),
    )
