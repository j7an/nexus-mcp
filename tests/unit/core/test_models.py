from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nexus_mcp.core.errors import InvalidJobTransitionError
from nexus_mcp.core.models import (
    ConfigLayerSnapshot,
    ExecutionConfigValues,
    JobState,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    WorkspaceSelector,
    validate_job_transition,
)


def test_workspace_selector_requires_exactly_one_identity():
    """Selectors cannot be ambiguous or identity-free."""
    with pytest.raises(ValidationError):
        WorkspaceSelector()
    with pytest.raises(ValidationError):
        WorkspaceSelector(workspace_id="ws", path=Path("/tmp/repo"))


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_terminal_job_state_cannot_reopen(terminal: JobState):
    """Terminal snapshots never return to an active state."""
    with pytest.raises(InvalidJobTransitionError):
        validate_job_transition(terminal, "running")


def test_requested_and_resolved_configuration_are_immutable():
    """Resolved configuration cannot change after execution begins."""
    requested = RequestedExecutionConfig(
        explicit=ExecutionConfigValues(model="model-a", timeout_seconds=30)
    )
    resolved = ResolvedExecutionConfig.from_requested(requested, backend_defaults={})

    with pytest.raises(ValidationError):
        resolved.model = "model-b"


def test_configuration_resolution_uses_explicit_provider_and_snapshotted_layers_in_order():
    """Higher-precedence values win while absent values retain their source."""
    requested = RequestedExecutionConfig(
        explicit=ExecutionConfigValues(timeout_seconds=10),
        workspace=ConfigLayerSnapshot(
            values=ExecutionConfigValues(model="workspace-model", timeout_seconds=20),
            source="/repo/.nexus/config.toml",
            source_hash="a" * 64,
        ),
        user=ConfigLayerSnapshot(
            values=ExecutionConfigValues(model="user-model", output_limit_bytes=4000),
            source="/Users/test/Library/Application Support/nexus-mcp/config.toml",
            source_hash="b" * 64,
        ),
        environment=ConfigLayerSnapshot(
            values=ExecutionConfigValues(model="environment-model", output_limit_bytes=2000),
            source="environment",
            source_hash="c" * 64,
        ),
    )

    resolved = ResolvedExecutionConfig.from_requested(
        requested,
        backend_defaults=ExecutionConfigValues(model="provider-model"),
    )

    assert resolved.timeout_seconds == 10
    assert resolved.model == "provider-model"
    assert resolved.output_limit_bytes == 4000
    assert resolved.sources == {
        "model": "provider",
        "timeout_seconds": "explicit",
        "output_limit_bytes": "user",
    }


def test_configuration_rejects_secret_bearing_fields():
    """Snapshots persist policy values, never credentials."""
    with pytest.raises(ValidationError):
        ExecutionConfigValues.model_validate({"model": "model-a", "api_key": "secret"})


def test_configuration_snapshot_requires_utc_timestamp():
    """Persisted core timestamps remain unambiguous across processes."""
    with pytest.raises(ValidationError):
        ConfigLayerSnapshot(
            values=ExecutionConfigValues(),
            source="environment",
            source_hash="a" * 64,
            captured_at=datetime(2026, 8, 30),
        )

    snapshot = ConfigLayerSnapshot(
        values=ExecutionConfigValues(),
        source="environment",
        source_hash="a" * 64,
        captured_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    assert snapshot.captured_at.tzinfo is UTC
