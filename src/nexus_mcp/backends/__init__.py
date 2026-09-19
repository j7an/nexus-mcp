"""Framework-independent backend protocols and runtime registry."""

from nexus_mcp.backends.base import (
    ActiveReconciliationOutcome,
    AgentBackend,
    BackendExecutionContext,
    BackendFailure,
    CancelledReconciliationOutcome,
    CancelRequested,
    CompletedReconciliationOutcome,
    ControlSignal,
    FailedReconciliationOutcome,
    InputRequiredReconciliationOutcome,
    InputResolved,
    LeaseLost,
    ReconciliationOutcome,
    RetryDisposition,
    RuntimeShutdown,
    UnknownReconciliationOutcome,
)
from nexus_mcp.backends.claude_agent import ClaudeAgentBackend
from nexus_mcp.backends.manager import BackendManager

__all__ = [
    "ActiveReconciliationOutcome",
    "AgentBackend",
    "BackendExecutionContext",
    "BackendFailure",
    "BackendManager",
    "CancelRequested",
    "CancelledReconciliationOutcome",
    "ClaudeAgentBackend",
    "CompletedReconciliationOutcome",
    "ControlSignal",
    "FailedReconciliationOutcome",
    "InputRequiredReconciliationOutcome",
    "InputResolved",
    "LeaseLost",
    "ReconciliationOutcome",
    "RetryDisposition",
    "RuntimeShutdown",
    "UnknownReconciliationOutcome",
]
