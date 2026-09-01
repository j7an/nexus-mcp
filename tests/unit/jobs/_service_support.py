"""Shared non-fixture helpers for application-service contract tests."""

from datetime import UTC, datetime, timedelta
from typing import Any

from nexus_mcp.core import (
    ConfigLayerSnapshot,
    ExecutionConfigValues,
    RequestedExecutionConfig,
    ReviewOperation,
    ReviewTarget,
    TurnOperation,
    WorkspaceSelector,
)
from nexus_mcp.jobs.service import AgentJobService
from tests.fixtures import make_access_context

NOW = datetime(2026, 8, 30, 20, 0, tzinfo=UTC)
WORKSPACE_SELECTOR = WorkspaceSelector(workspace_id="ws-test")


def authorized_access(**overrides: Any):
    """Create a caller explicitly trusted for the representative workspace."""
    defaults = {"authorized_workspace_ids": frozenset({"ws-test"})}
    return make_access_context(**(defaults | overrides))


def make_review_operation(**overrides: Any) -> ReviewOperation:
    """Create a representative inline working-tree review."""
    defaults = {"target": ReviewTarget(kind="working_tree"), "delivery": "inline"}
    return ReviewOperation(**(defaults | overrides))


async def _source_session(service: AgentJobService) -> str:
    """Create and terminalize one source session for idempotency tests."""
    source = await service.start(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        backend_id="codex",
        operation=TurnOperation(prompt="Establish source"),
        explicit_config=ExecutionConfigValues(),
        idempotency_key="source",
    )
    assert source.session_id is not None
    await service.cancel(
        workspace=WORKSPACE_SELECTOR,
        access=authorized_access(),
        job_id=source.job_id,
    )
    return source.session_id


class _ChangingCaptureResolver:
    """Return semantically equal lower config through changing capture metadata."""

    def __init__(self) -> None:
        self._capture = 0

    def snapshot(
        self,
        backend_id: str,
        workspace,
        explicit: ExecutionConfigValues,
    ) -> RequestedExecutionConfig:
        self._capture += 1
        return RequestedExecutionConfig(
            explicit=explicit,
            workspace=ConfigLayerSnapshot(
                values=ExecutionConfigValues(timeout_seconds=30),
                source=f"workspace-{self._capture}",
                source_hash=f"{self._capture:064x}",
                captured_at=NOW + timedelta(seconds=self._capture),
            ),
        )


class _StableCaptureResolver:
    """Capture only explicit values so checkpoint changes are isolated."""

    def snapshot(
        self,
        backend_id: str,
        workspace,
        explicit: ExecutionConfigValues,
    ) -> RequestedExecutionConfig:
        return RequestedExecutionConfig(explicit=explicit)
