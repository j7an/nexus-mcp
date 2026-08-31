import json
import operator
from collections.abc import Mapping

import pytest
from pydantic import TypeAdapter, ValidationError

from nexus_mcp.core.capabilities import BackendCapabilities
from nexus_mcp.core.models import AgentJob
from nexus_mcp.core.operations import (
    AgentOperation,
    ForkOperation,
    ReviewOperation,
    TurnOperation,
)
from tests.fixtures import make_agent_job


def test_operation_union_round_trips_turn():
    """Removing the turn discriminator would break persisted-operation decoding."""
    value = TypeAdapter(AgentOperation).validate_python({"kind": "turn", "prompt": "Inspect"})

    assert isinstance(value, TurnOperation)
    assert value.prompt == "Inspect"


def test_operation_union_rejects_unknown_kind():
    """Provider-specific commands cannot silently extend the core operation union."""
    with pytest.raises(ValidationError):
        TypeAdapter(AgentOperation).validate_python({"kind": "provider_command"})


def test_operation_union_round_trips_a_pure_fork():
    """Forking a checkpoint does not require an unrelated turn prompt."""
    value = TypeAdapter(AgentOperation).validate_python({"kind": "fork"})

    assert isinstance(value, ForkOperation)


def test_turn_context_is_deeply_immutable_and_serializes_as_json():
    """Mutable request input cannot alter an admitted operation snapshot."""
    operation = TurnOperation(
        prompt="Inspect",
        context={"metadata": {"priority": "high"}, "paths": ["src"]},
    )
    metadata = operation.context["metadata"]
    paths = operation.context["paths"]
    assert isinstance(metadata, Mapping)

    with pytest.raises(TypeError):
        operator.setitem(operation.context, "later", True)
    with pytest.raises(TypeError):
        operator.setitem(metadata, "priority", "low")
    with pytest.raises(TypeError):
        operator.setitem(paths, 0, "tests")

    assert json.loads(operation.model_dump_json())["context"] == {
        "metadata": {"priority": "high"},
        "paths": ["src"],
    }


def test_turn_context_enforces_exact_canonical_json_byte_limit():
    """Nested context cannot exceed the one MiB canonical persistence boundary."""
    TurnOperation(prompt="Inspect", context={"payload": "x" * 1_048_562})

    with pytest.raises(ValidationError):
        TurnOperation(prompt="Inspect", context={"payload": "x" * 1_048_563})


def test_turn_context_rejects_nesting_deeper_than_32():
    """Adversarial context cannot exceed the recursive validation depth."""
    within_limit: object = "leaf"
    for _ in range(31):
        within_limit = [within_limit]
    TurnOperation(prompt="Inspect", context={"nested": within_limit})

    beyond_limit: object = "leaf"
    for _ in range(32):
        beyond_limit = [beyond_limit]
    with pytest.raises(ValidationError):
        TurnOperation(prompt="Inspect", context={"nested": beyond_limit})


def test_turn_context_rejects_nested_container_above_4096_items():
    """A small outer request cannot hide an unbounded nested sequence."""
    TurnOperation(prompt="Inspect", context={"items": [0] * 4096})

    with pytest.raises(ValidationError):
        TurnOperation(prompt="Inspect", context={"items": [0] * 4097})


def test_review_operation_defaults_to_inline_and_round_trips_delivery():
    """Review delivery remains explicit in persisted operations, including the default."""
    inline = ReviewOperation(target={"kind": "working_tree"})
    detached = TypeAdapter(AgentOperation).validate_python(
        {
            "kind": "review",
            "target": {"kind": "commit", "reference": "abc123"},
            "delivery": "detached",
        }
    )

    assert inline.delivery == "inline"
    assert isinstance(detached, ReviewOperation)
    assert detached.delivery == "detached"
    assert detached.target.reference == "abc123"


def test_backend_capabilities_advertise_review_delivery_and_dormant_structured_output():
    """Admission can distinguish supported review delivery without inventing a request schema."""
    capabilities = BackendCapabilities(
        operations=frozenset({"review"}),
        review_deliveries=frozenset({"inline", "detached"}),
    )

    assert capabilities.review_deliveries == frozenset({"inline", "detached"})
    assert capabilities.structured_output is False


def test_agent_job_keeps_a_typed_operation():
    """Durable jobs cannot regress to an untyped provider dictionary."""
    job = make_agent_job(operation=TurnOperation(prompt="Inspect"))

    assert isinstance(job, AgentJob)
    assert isinstance(job.operation, TurnOperation)
