import json
import operator

import pytest
from pydantic import TypeAdapter, ValidationError

from nexus_mcp.core.operations import ReviewTarget
from nexus_mcp.core.results import (
    CancelReceipt,
    JobError,
    JobResultResponse,
    OperationResult,
    ReviewResult,
    TurnResult,
)
from tests.fixtures import make_job_error, make_turn_result


def test_review_result_includes_inconclusive_in_verdict_union():
    """Incomplete provider state remains inconclusive rather than becoming a false pass."""
    result = ReviewResult(
        verdict="inconclusive",
        summary="Provider state was incomplete",
        target=ReviewTarget(kind="working_tree"),
        delivery="inline",
    )

    assert result.verdict == "inconclusive"


def test_review_result_round_trips_target_and_delivery():
    """Review scope and delivery survive the normalized result boundary."""
    result = ReviewResult(
        verdict="pass",
        summary="No blocking findings",
        target=ReviewTarget(kind="commit", reference="abc123"),
        delivery="detached",
    )

    decoded = TypeAdapter(OperationResult).validate_json(result.model_dump_json())

    assert isinstance(decoded, ReviewResult)
    assert decoded.target == ReviewTarget(kind="commit", reference="abc123")
    assert decoded.delivery == "detached"


def test_job_error_rejects_unbounded_provider_payload():
    """Provider diagnostics cannot bypass the durable byte limit."""
    with pytest.raises(ValidationError):
        JobError(code="provider_failed", message="failed", details={"raw": "x" * 20_000})


def test_job_error_details_are_deeply_immutable_and_serialize_as_json():
    """Normalized diagnostics cannot mutate after validation or leak read-only containers."""
    error = make_job_error(details={"provider": {"status": "failed"}, "attempts": [1, 2]})
    provider = error.details["provider"]
    attempts = error.details["attempts"]

    with pytest.raises(TypeError):
        operator.setitem(error.details, "later", True)
    with pytest.raises(TypeError):
        operator.setitem(provider, "status", "ok")
    with pytest.raises(TypeError):
        operator.setitem(attempts, 0, 3)

    assert json.loads(error.model_dump_json())["details"] == {
        "attempts": [1, 2],
        "provider": {"status": "failed"},
    }


def test_structured_output_rejects_nested_payload_above_one_mibibyte():
    """Optional structured output cannot evade the normalized JSON byte cap."""
    with pytest.raises(ValidationError):
        TurnResult(message="done", structured_output={"payload": "x" * 1_048_563})


def test_turn_usage_rejects_nesting_deeper_than_32():
    """Small usage metadata cannot hide pathologically deep normalized JSON."""
    nested: object = "leaf"
    for _ in range(32):
        nested = [nested]

    with pytest.raises(ValidationError):
        TurnResult(message="done", usage={"nested": nested})


def test_job_error_rejects_nested_container_above_4096_items():
    """Bounded diagnostic bytes also enforce the per-container item limit."""
    with pytest.raises(ValidationError):
        JobError(code="provider_failed", message="failed", details={"items": [0] * 4097})


def test_job_result_response_discriminates_succeeded_payload():
    """Public result polling preserves the terminal response variant and typed payload."""
    response = TypeAdapter(JobResultResponse).validate_python(
        {
            "status": "succeeded",
            "job_id": "job-test",
            "result": {
                "job_id": "job-test",
                "payload": {"kind": "turn", "message": "done"},
            },
        }
    )

    assert response.status == "succeeded"
    assert isinstance(response.result.payload, TurnResult)


def test_result_fixture_accepts_explicit_overrides():
    """Core test factories retain stable defaults without blocking targeted values."""
    result = make_turn_result(message="custom")

    assert result.message == "custom"


def test_cancel_receipt_reports_whether_this_call_committed_an_event():
    """Callers need committed-event truth to avoid notifier wakes for atomic no-ops."""
    receipt = CancelReceipt(
        job_id="job-test",
        state="running",
        cancel_requested=True,
        completed_immediately=False,
        event_committed=True,
    )

    assert receipt.event_committed is True
