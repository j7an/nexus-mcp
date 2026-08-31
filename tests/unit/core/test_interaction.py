import pytest
from pydantic import TypeAdapter, ValidationError

from nexus_mcp.core.interaction import (
    ApprovalRequest,
    ApprovalResponse,
    InputRequest,
    PendingInput,
    PermissionRequest,
    PermissionResponse,
)
from tests.fixtures import make_pending_permission


def test_input_request_union_round_trips_permission():
    """Persisted input requests retain their normalized permission variant."""
    request = TypeAdapter(InputRequest).validate_python(
        {
            "kind": "permission",
            "prompt": "Allow network access?",
            "requested": ["network:api.example.com"],
        }
    )

    assert isinstance(request, PermissionRequest)


def test_permission_response_must_be_subset_of_requested_permissions():
    """A response cannot grant authority absent from the provider request."""
    pending = make_pending_permission(requested=["network:api.example.com"])

    with pytest.raises(ValidationError):
        pending.validate_response(PermissionResponse(granted=["filesystem:/"]))


def test_permission_response_returns_normalized_model():
    """Valid permission responses remain provider-neutral typed values."""
    pending = make_pending_permission(
        requested=["network:api.example.com", "filesystem:/workspace"]
    )

    response = pending.validate_response(PermissionResponse(granted=["network:api.example.com"]))

    assert isinstance(response, PermissionResponse)
    assert response.granted == frozenset({"network:api.example.com"})


def test_approval_response_must_use_an_allowed_decision():
    """A caller cannot choose an approval outcome the request did not offer."""
    pending = PendingInput(
        input_id="input-test",
        job_id="job-test",
        request=ApprovalRequest(
            prompt="Run the command?",
            allowed_decisions=frozenset({"deny"}),
        ),
    )

    with pytest.raises(ValidationError):
        pending.validate_response(ApprovalResponse(decision="approve"))


def test_pending_input_rejects_response_for_another_request_kind():
    """Response dispatch cannot confuse approval and permission shapes."""
    pending = make_pending_permission(requested=["network:api.example.com"])

    with pytest.raises(ValidationError):
        pending.validate_response(ApprovalResponse(decision="deny"))
