"""Scripted backend doubles for worker and lifecycle tests."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from nexus_mcp.backends import (
    BackendExecutionContext,
    BackendFailure,
    ReconciliationOutcome,
)
from nexus_mcp.core import (
    AgentOperation,
    BackendAvailability,
    BackendCapabilities,
    BackendDescriptor,
    BackendEvent,
    InputRequest,
    OperationKind,
    OperationResult,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    Workspace,
)

__all__ = [
    "EmitEventAction",
    "RaiseFailureAction",
    "RecordReferenceAction",
    "RequestInputAction",
    "ReturnReconciliationAction",
    "ReturnResultAction",
    "ScriptedBackend",
]


@dataclass(frozen=True)
class EmitEventAction:
    """Tell a scripted backend to emit one normalized event."""

    event: BackendEvent


@dataclass(frozen=True)
class RecordReferenceAction:
    """Tell a scripted backend to record one provider reference."""

    reference: ProviderReference


@dataclass(frozen=True)
class RequestInputAction:
    """Tell a scripted backend to request one normalized input."""

    request: InputRequest


@dataclass(frozen=True)
class RaiseFailureAction:
    """Tell a scripted backend to raise one classified backend failure."""

    failure: BackendFailure


@dataclass(frozen=True)
class ReturnResultAction:
    """Tell a scripted backend to return one normalized operation result."""

    result: OperationResult


@dataclass(frozen=True)
class ReturnReconciliationAction:
    """Tell a scripted backend to return one reconciliation decision."""

    outcome: ReconciliationOutcome


type ContextAction = EmitEventAction | RecordReferenceAction | RequestInputAction
type ExecutionAction = ContextAction | RaiseFailureAction | ReturnResultAction
type ReconciliationAction = ContextAction | RaiseFailureAction | ReturnReconciliationAction


class ScriptedBackend:
    """A deterministic backend whose externally visible effects are supplied by test scripts."""

    def __init__(self, backend_id: str = "scripted") -> None:
        self.descriptor = BackendDescriptor(
            backend_id=backend_id,
            display_name="Scripted Backend",
            capabilities=BackendCapabilities(
                operations=frozenset({"turn", "fork", "review", "diagnostics"}),
                cancellation=True,
                graceful_interrupt=True,
                session_fork=True,
                input_required=True,
            ),
        )
        self.availability = BackendAvailability(available=True)
        self.execute_calls: list[tuple[AgentOperation, BackendExecutionContext]] = []
        self.reconcile_calls: list[
            tuple[tuple[ProviderReference, ...], BackendExecutionContext]
        ] = []
        self.config_calls: list[tuple[RequestedExecutionConfig, Workspace]] = []
        self.close_calls = 0
        self._execution_actions: deque[ExecutionAction] = deque()
        self._reconciliation_actions: deque[ReconciliationAction] = deque()

    @property
    def backend_id(self) -> str:
        """Return the stable registered identifier."""
        return self.descriptor.backend_id

    def with_operations(self, operations: Iterable[OperationKind]) -> Self:
        """Replace advertised operations while preserving every other static capability."""
        self.descriptor = self.descriptor.model_copy(
            update={
                "capabilities": self.descriptor.capabilities.model_copy(
                    update={"operations": frozenset(operations)}
                )
            }
        )
        return self

    def queue_execute(self, *actions: ExecutionAction) -> None:
        """Append exact effects and one eventual result or failure for ``execute``."""
        self._execution_actions.extend(actions)

    def queue_reconcile(self, *actions: ReconciliationAction) -> None:
        """Append exact effects and one eventual outcome or failure for ``reconcile``."""
        self._reconciliation_actions.extend(actions)

    async def check_availability(self, workspace: Workspace) -> BackendAvailability:
        """Return the configured health observation without mutating registration state."""
        return self.availability

    async def resolve_execution_config(
        self, requested: RequestedExecutionConfig, workspace: Workspace
    ) -> ResolvedExecutionConfig:
        """Record and resolve the requested configuration with no provider defaults."""
        self.config_calls.append((requested, workspace))
        return ResolvedExecutionConfig.from_requested(requested, backend_defaults={})

    async def execute(
        self, operation: AgentOperation, context: BackendExecutionContext
    ) -> OperationResult:
        """Apply scripted effects until one exact result or classified failure is reached."""
        self.execute_calls.append((operation, context))
        while self._execution_actions:
            action = self._execution_actions.popleft()
            match action:
                case EmitEventAction() | RecordReferenceAction() | RequestInputAction():
                    await self._apply_context_action(action, context)
                case RaiseFailureAction(failure=failure):
                    raise failure
                case ReturnResultAction(result=result):
                    return result
        raise AssertionError("Scripted backend execute() has no queued outcome")

    async def reconcile(
        self,
        provider_state: tuple[ProviderReference, ...],
        context: BackendExecutionContext,
    ) -> ReconciliationOutcome:
        """Apply scripted effects until one exact reconciliation outcome or failure is reached."""
        self.reconcile_calls.append((provider_state, context))
        while self._reconciliation_actions:
            action = self._reconciliation_actions.popleft()
            match action:
                case EmitEventAction() | RecordReferenceAction() | RequestInputAction():
                    await self._apply_context_action(action, context)
                case RaiseFailureAction(failure=failure):
                    raise failure
                case ReturnReconciliationAction(outcome=outcome):
                    return outcome
        raise AssertionError("Scripted backend reconcile() has no queued outcome")

    async def close(self) -> None:
        """Record one runtime close request."""
        self.close_calls += 1

    async def _apply_context_action(
        self, action: ContextAction, context: BackendExecutionContext
    ) -> None:
        """Apply one shared context effect without accessing worker-owned state directly."""
        match action:
            case EmitEventAction(event=event):
                await context.emit(event)
            case RecordReferenceAction(reference=reference):
                await context.record_provider_reference(reference)
            case RequestInputAction(request=request):
                await context.request_input(request)
