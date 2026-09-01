import json
import operator
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from nexus_mcp.core.errors import InvalidJobTransitionError
from nexus_mcp.core.interaction import PermissionResponse
from nexus_mcp.core.models import (
    BackendEvent,
    ConfigLayerSnapshot,
    ExecutionConfigValues,
    JobEvent,
    JobState,
    JobStatus,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    RetryPolicy,
    WorkspaceSelector,
    validate_job_transition,
)
from tests.fixtures import make_agent_job, make_pending_permission


@pytest.fixture(params=["backend", "committed"])
def event_with_nested_payload(request: pytest.FixtureRequest) -> BackendEvent | JobEvent:
    """Build each event boundary with mutable JSON input containers."""
    payload = {"details": {"status": "running"}, "steps": ["first", "second"]}
    if request.param == "backend":
        return BackendEvent(type="progress", payload=payload)
    return JobEvent(job_id="job-test", sequence=1, type="progress", payload=payload)


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


def test_event_payloads_are_deeply_immutable(
    event_with_nested_payload: BackendEvent | JobEvent,
):
    """Committed event content cannot change through nested JSON containers."""
    details = event_with_nested_payload.payload["details"]
    steps = event_with_nested_payload.payload["steps"]
    assert isinstance(details, Mapping)

    with pytest.raises(TypeError):
        operator.setitem(event_with_nested_payload.payload, "message", "changed")
    with pytest.raises(TypeError):
        operator.setitem(details, "status", "changed")
    with pytest.raises(TypeError):
        operator.setitem(steps, 0, "changed")


def test_event_payloads_preserve_json_dump_shape(
    event_with_nested_payload: BackendEvent | JobEvent,
):
    """Immutable event payloads still dump as ordinary JSON objects and arrays."""
    expected = {"details": {"status": "running"}, "steps": ["first", "second"]}

    assert event_with_nested_payload.model_dump()["payload"] == expected
    assert json.loads(event_with_nested_payload.model_dump_json())["payload"] == expected


@pytest.mark.parametrize("committed", [False, True], ids=["backend", "job"])
def test_event_payload_rejects_more_than_one_mibibyte(committed: bool):
    """Normalized event envelopes cannot bypass the shared canonical byte ceiling."""
    kwargs = {"job_id": "job-test", "sequence": 1} if committed else {}
    event_type = JobEvent if committed else BackendEvent

    with pytest.raises(ValidationError, match="canonical UTF-8 bytes"):
        event_type(type="log", payload={"message": "x" * 1_048_576}, **kwargs)


@pytest.mark.parametrize("committed", [False, True], ids=["backend", "job"])
def test_event_payload_rejects_nesting_deeper_than_32(committed: bool):
    """Small event payloads cannot hide pathologically deep provider structures."""
    nested: object = "leaf"
    for _ in range(32):
        nested = [nested]
    kwargs = {"job_id": "job-test", "sequence": 1} if committed else {}
    event_type = JobEvent if committed else BackendEvent

    with pytest.raises(ValidationError, match="maximum nesting depth"):
        event_type(type="log", payload={"nested": nested}, **kwargs)


@pytest.mark.parametrize("committed", [False, True], ids=["backend", "job"])
def test_event_payload_rejects_container_above_4096_items(committed: bool):
    """One nested event container cannot grow without the shared item ceiling."""
    kwargs = {"job_id": "job-test", "sequence": 1} if committed else {}
    event_type = JobEvent if committed else BackendEvent

    with pytest.raises(ValidationError, match="maximum item count"):
        event_type(type="log", payload={"items": [0] * 4097}, **kwargs)


def test_resolved_configuration_sources_are_immutable():
    """Resolved provenance cannot change after execution begins."""
    requested = RequestedExecutionConfig(explicit=ExecutionConfigValues(model="model-a"))
    resolved = ResolvedExecutionConfig.from_requested(requested, backend_defaults={})

    with pytest.raises(TypeError):
        operator.setitem(resolved.sources, "model", "fallback")


def test_resolved_configuration_sources_dump_as_plain_dict():
    """Immutable provenance retains Pydantic's ordinary dictionary dump contract."""
    requested = RequestedExecutionConfig(explicit=ExecutionConfigValues(model="model-a"))
    resolved = ResolvedExecutionConfig.from_requested(requested, backend_defaults={})

    assert resolved.model_dump()["sources"] == {"model": "explicit"}
    assert json.loads(resolved.model_dump_json())["sources"] == {"model": "explicit"}


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


def test_configuration_resolution_labels_authorized_legacy_fallbacks():
    """Legacy Nexus defaults remain distinguishable from provider-native defaults."""
    resolved = ResolvedExecutionConfig.from_requested(
        RequestedExecutionConfig(),
        backend_defaults=ExecutionConfigValues(),
        fallback_defaults=ExecutionConfigValues(model="legacy-model"),
        fallback_source="legacy_nexus_fallback",
    )

    assert resolved.model == "legacy-model"
    assert resolved.sources == {"model": "legacy_nexus_fallback"}


def test_configuration_rejects_secret_bearing_fields():
    """Snapshots persist policy values, never credentials."""
    with pytest.raises(ValidationError):
        ExecutionConfigValues.model_validate({"model": "model-a", "api_key": "secret"})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("base_delay_seconds", float("inf")),
        ("base_delay_seconds", float("-inf")),
        ("base_delay_seconds", float("nan")),
        ("max_delay_seconds", float("inf")),
        ("max_delay_seconds", float("-inf")),
        ("max_delay_seconds", float("nan")),
    ],
)
def test_retry_policy_rejects_non_finite_delays(field_name: str, value: float):
    """Retry scheduling admits only finite non-negative delay bounds."""
    with pytest.raises(ValidationError):
        RetryPolicy.model_validate({field_name: value})


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


def test_job_status_projects_authoritative_execution_state_and_serializes():
    """Public status retains phase, input, effective config, and event progress."""
    resolved = ResolvedExecutionConfig.from_requested(
        RequestedExecutionConfig(explicit=ExecutionConfigValues(model="model-a")),
        backend_defaults={},
    )
    job = make_agent_job(
        state="input_required",
        resolved_config=resolved,
        cancel_requested_at=datetime(2026, 8, 30, tzinfo=UTC),
    )
    pending = make_pending_permission()

    status = JobStatus.from_job(
        job,
        phase="executing",
        pending_inputs=(pending,),
        latest_event_sequence=7,
    )
    dumped = json.loads(status.model_dump_json())

    assert status.phase == "executing"
    assert status.pending_inputs == (pending,)
    assert status.resolved_config == resolved
    assert status.latest_event_sequence == 7
    assert status.cancel_requested is True
    assert dumped["pending_inputs"][0]["request"]["kind"] == "permission"
    assert dumped["resolved_config"]["model"] == "model-a"
    assert dumped["latest_event_sequence"] == 7
    assert dumped["cancel_requested"] is True

    with pytest.raises(TypeError):
        operator.setitem(status.pending_inputs, 0, pending)


def test_job_status_rejects_negative_latest_event_sequence():
    """Authoritative event progress cannot precede sequence zero."""
    with pytest.raises(ValidationError):
        JobStatus.from_job(make_agent_job(), latest_event_sequence=-1)


def test_job_status_pending_inputs_must_be_unresolved():
    """Resolved interaction records cannot appear in the unresolved status collection."""
    resolved_input = make_pending_permission(
        response=PermissionResponse(granted=["network:api.example.com"])
    )

    with pytest.raises(ValidationError):
        JobStatus.from_job(make_agent_job(), pending_inputs=(resolved_input,))
