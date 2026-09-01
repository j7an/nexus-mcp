"""Canonical application-service admission, access, query, and control behavior."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from nexus_mcp.backends import BackendManager
from nexus_mcp.core import (
    AccessDeniedError,
    BackendEvent,
    ExecutionConfigValues,
    ForkOperation,
    IdempotencyConflictError,
    ProviderReference,
    TurnOperation,
    UnsupportedCapabilityError,
)
from nexus_mcp.jobs.events import EventNotifier
from nexus_mcp.jobs.service import AgentJobService
from tests.fixtures import (
    make_agent_session,
)
from tests.unit.jobs._service_support import (
    NOW,
    WORKSPACE_SELECTOR,
    _source_session,
    _StableCaptureResolver,
    authorized_access,
    make_review_operation,
)

if TYPE_CHECKING:
    from nexus_mcp.jobs.configuration import NexusConfigResolver


async def test_source_session_workspace_mismatch_is_access_denied(
    service: AgentJobService,
    store: Mock,
):
    """A session cannot be continued through a different durable workspace selector."""
    store.get_session.return_value = make_agent_session(workspace_id="ws-other")

    with pytest.raises(AccessDeniedError):
        await service.continue_session(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            session_id="session-test",
            operation=TurnOperation(prompt="Continue"),
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()


async def test_fork_creates_child_from_parent_checkpoint(
    service: AgentJobService,
    store: Mock,
):
    """A fork gets a new session while idempotency and provider state derive from its parent."""
    operation = ForkOperation(prompt="Try another approach")

    handle = await service.fork_session(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        session_id="session-test",
        operation=operation,
        explicit_config=ExecutionConfigValues(),
    )

    command = store.create_job.await_args.args[0]
    assert handle.session_id == command.session_id
    assert command.session_id != "session-test"
    assert command.create_session is True
    assert command.parent_session_id == "session-test"
    assert command.source_session_id == "session-test"
    assert command.source_checkpoint == (ProviderReference(kind="thread", value="thread-test"),)
    assert command.command_family == "fork_session"


@pytest.mark.parametrize(
    ("delivery", "create_session", "parent_session_id"),
    [("inline", False, None), ("detached", True, "session-test")],
)
async def test_review_delivery_selects_existing_or_child_session_semantics(
    service: AgentJobService,
    store: Mock,
    delivery: str,
    create_session: bool,
    parent_session_id: str | None,
):
    """Inline review continues the source while detached review inherits into a child."""
    operation = make_review_operation(delivery=delivery)

    handle = await service.review(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        session_id="session-test",
        operation=operation,
        explicit_config=ExecutionConfigValues(),
    )

    command = store.create_job.await_args.args[0]
    assert command.create_session is create_session
    assert command.parent_session_id == parent_session_id
    assert command.source_session_id == "session-test"
    assert command.source_checkpoint == (ProviderReference(kind="thread", value="thread-test"),)
    if delivery == "inline":
        assert handle.session_id == "session-test"
    else:
        assert handle.session_id == command.session_id
        assert command.session_id != "session-test"


async def test_continue_uses_existing_session_policy_owner_and_checkpoint(
    service: AgentJobService,
    store: Mock,
):
    """Continuation must not replace the durable source session or broaden its identity."""
    store.get_session.return_value = make_agent_session(
        owner_id="local:400", access_policy="workspace"
    )
    operation = TurnOperation(prompt="Continue")

    handle = await service.continue_session(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(principal_id="local:501"),
        session_id="session-test",
        operation=operation,
        explicit_config=ExecutionConfigValues(),
        idempotency_key="continue-1",
    )

    command = store.create_job.await_args.args[0]
    assert handle.session_id == "session-test"
    assert command.session_id == "session-test"
    assert command.create_session is False
    assert command.parent_session_id is None
    assert command.source_session_id == "session-test"
    assert command.source_checkpoint == (ProviderReference(kind="thread", value="thread-test"),)
    assert command.owner_id == "local:400"
    assert command.access_policy == "workspace"
    assert command.command_family == "continue_session"
    assert command.idempotency_key == "continue-1"


async def test_continue_rejects_backend_without_session_continuation_before_job_creation(
    service: AgentJobService,
    store: Mock,
    backend: Mock,
):
    """A fresh legacy invocation cannot masquerade as continuation of provider state."""
    backend.descriptor = backend.descriptor.model_copy(
        update={
            "capabilities": backend.descriptor.capabilities.model_copy(
                update={"session_continuation": False}
            )
        }
    )

    with pytest.raises(UnsupportedCapabilityError, match="session_continuation"):
        await service.continue_session(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(),
            session_id="session-test",
            operation=TurnOperation(prompt="Continue"),
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()


async def test_private_session_creation_access_violation_is_access_denied(
    service: AgentJobService,
    store: Mock,
):
    """Creation access failures stay explicit instead of disclosing via read-style not-found."""
    store.get_session.return_value = make_agent_session(owner_id="local:400")

    with pytest.raises(AccessDeniedError):
        await service.continue_session(
            workspace=WORKSPACE_SELECTOR,
            access=authorized_access(principal_id="local:501"),
            session_id="session-test",
            operation=TurnOperation(prompt="Continue"),
            explicit_config=ExecutionConfigValues(),
        )

    store.create_job.assert_not_awaited()


@pytest.mark.parametrize(
    "family",
    ["start", "continue", "fork", "review_inline", "review_detached", "diagnose"],
)
async def test_semantic_idempotency_replays_every_service_command_family(
    real_service_environment,
    family: str,
):
    """Generated identities and lower capture metadata cannot conflict with the same intent."""
    service, _, _, _ = real_service_environment
    common = {
        "workspace": WORKSPACE_SELECTOR,
        "access": authorized_access(),
        "explicit_config": ExecutionConfigValues(model="gpt-5"),
        "idempotency_key": f"replay-{family}",
    }
    source_session_id = None
    if family not in {"start", "diagnose"}:
        source_session_id = await _source_session(service)

    async def invoke():
        match family:
            case "start":
                return await service.start(
                    **common,
                    backend_id="codex",
                    operation=TurnOperation(prompt="Inspect"),
                )
            case "diagnose":
                return await service.diagnose(**common, backend_id="codex")
            case "continue":
                return await service.continue_session(
                    **common,
                    session_id=source_session_id,
                    operation=TurnOperation(prompt="Continue"),
                )
            case "fork":
                return await service.fork_session(
                    **common,
                    session_id=source_session_id,
                    operation=ForkOperation(prompt="Fork"),
                )
            case "review_inline" | "review_detached":
                delivery = "inline" if family == "review_inline" else "detached"
                return await service.review(
                    **common,
                    session_id=source_session_id,
                    operation=make_review_operation(delivery=delivery),
                )
            case _:
                raise AssertionError(f"unknown command family: {family}")

    first = await invoke()
    replay = await invoke()

    assert replay == first


async def test_semantic_idempotency_still_conflicts_on_explicit_intent_change(
    real_service_environment,
):
    """Excluding persistence captures must not collapse two different caller prompts."""
    service, _, _, _ = real_service_environment
    common = {
        "workspace": WORKSPACE_SELECTOR,
        "access": authorized_access(),
        "backend_id": "codex",
        "explicit_config": ExecutionConfigValues(),
        "idempotency_key": "conflict",
    }
    await service.start(**common, operation=TurnOperation(prompt="First"))

    with pytest.raises(IdempotencyConflictError):
        await service.start(**common, operation=TurnOperation(prompt="Second"))


async def test_semantic_idempotency_replays_after_terminal_completion(
    real_service_environment,
):
    """A semantic retry returns its original handle even after that job terminalizes."""
    service, _, _, _ = real_service_environment
    kwargs = {
        "workspace": WORKSPACE_SELECTOR,
        "access": authorized_access(),
        "backend_id": "codex",
        "operation": TurnOperation(prompt="Terminal replay"),
        "explicit_config": ExecutionConfigValues(),
        "idempotency_key": "terminal-replay",
    }
    first = await service.start(**kwargs)
    await service.cancel(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id=first.job_id,
    )

    replay = await service.start(**kwargs)

    assert replay == first


async def test_semantic_idempotency_ignores_recaptured_source_checkpoints(
    real_service_environment,
):
    """A later valid provider checkpoint must not change the caller's fork intent."""
    service, durable_store, backend, _ = real_service_environment
    source = await service.start(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        backend_id="codex",
        operation=TurnOperation(prompt="Checkpoint source"),
        explicit_config=ExecutionConfigValues(),
    )
    assert source.session_id is not None
    claimed = await durable_store.claim_next(
        "worker-checkpoint",
        datetime(2099, 1, 1, tzinfo=UTC),
        event=BackendEvent(type="progress", occurred_at=NOW),
    )
    assert claimed is not None and claimed.job.job_id == source.job_id
    first_reference = ProviderReference(kind="thread", value="thread-first")
    second_reference = ProviderReference(kind="turn", value="turn-second")
    await durable_store.record_provider_reference(claimed.token, first_reference)
    await durable_store.record_provider_reference(claimed.token, second_reference)
    await service.cancel(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id=source.job_id,
    )
    checkpoint_calls = 0

    async def changing_checkpoint(
        *,
        session_id: str | None = None,
        job_id: str | None = None,
    ) -> tuple[ProviderReference, ...]:
        nonlocal checkpoint_calls
        assert session_id == source.session_id and job_id is None
        checkpoint_calls += 1
        if checkpoint_calls == 1:
            return (first_reference,)
        return (first_reference, second_reference)

    durable_store.get_provider_references = changing_checkpoint  # type: ignore[method-assign]
    stable_service = AgentJobService(
        store=durable_store,
        backend_manager=BackendManager([backend]),
        config_resolver=cast("NexusConfigResolver", _StableCaptureResolver()),
        notifier=EventNotifier(),
    )
    kwargs = {
        "workspace": WORKSPACE_SELECTOR,
        "access": authorized_access(),
        "session_id": source.session_id,
        "operation": ForkOperation(prompt="Fork from checkpoint"),
        "explicit_config": ExecutionConfigValues(),
        "idempotency_key": "checkpoint-replay",
    }

    first = await stable_service.fork_session(**kwargs)
    replay = await stable_service.fork_session(**kwargs)

    assert replay == first
