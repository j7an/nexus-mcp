"""Durable job runtime ownership for FastMCP lifespan and direct tool calls."""

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass, field
from pathlib import Path

from nexus_mcp.backends import BackendManager
from nexus_mcp.core import Workspace, WorkspaceInvalidError, WorkspaceSelector
from nexus_mcp.jobs import (
    AgentJobService,
    EventNotifier,
    EventPollingPolicy,
    ExponentialRetryDelay,
    NexusConfigResolver,
    RetryDelay,
    WorkerPolicy,
    WorkerPool,
)
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore
from nexus_mcp.legacy import legacy_backends

__all__ = ["MCPRuntime", "RuntimeProvider", "RuntimeTuning", "runtime_provider"]


@dataclass(frozen=True, slots=True)
class RuntimeTuning:
    """Explicit runtime timing dependencies, with production-safe defaults."""

    worker_policy: WorkerPolicy = field(default_factory=WorkerPolicy)
    event_polling_policy: EventPollingPolicy = field(default_factory=EventPollingPolicy)
    retry_delay: RetryDelay = field(default_factory=ExponentialRetryDelay)


class _MCPJobStore(SQLiteJobStore):
    """Admit a first explicit local path before its workspace row exists."""

    async def resolve_workspace(self, selector: WorkspaceSelector) -> Workspace:
        try:
            return await super().resolve_workspace(selector)
        except WorkspaceInvalidError:
            if selector.path is None:
                raise
        canonical_path = _canonical_workspace_path(selector.path)
        identity_path = os.path.normcase(str(canonical_path))
        workspace_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"nexus-mcp:local-workspace:{identity_path}")
        )
        return Workspace(workspace_id=workspace_id, canonical_path=canonical_path)


def _canonical_workspace_path(path: Path) -> Path:
    try:
        canonical = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WorkspaceInvalidError(str(path), "workspace path does not exist") from error
    if not canonical.is_dir():
        raise WorkspaceInvalidError(str(path), "workspace path is not a directory")
    return canonical


@dataclass(frozen=True, slots=True)
class MCPRuntime:
    """One opened durable-job object graph and its worker lifecycle."""

    store: SQLiteJobStore
    backends: BackendManager
    service: AgentJobService
    workers: WorkerPool
    event_polling_policy: EventPollingPolicy

    @classmethod
    @asynccontextmanager
    async def open(cls, tuning: RuntimeTuning | None = None) -> AsyncIterator["MCPRuntime"]:
        """Open all runtime resources and close them in strict reverse order."""
        effective_tuning = tuning or RuntimeTuning()
        async with AsyncExitStack() as stack:
            store = _MCPJobStore()
            await store.open()
            stack.push_async_callback(store.close)

            backends = BackendManager(legacy_backends())
            stack.push_async_callback(backends.close)
            notifier = EventNotifier()
            service = AgentJobService(
                store=store,
                backend_manager=backends,
                config_resolver=NexusConfigResolver(),
                notifier=notifier,
            )
            workers = WorkerPool(
                store=store,
                backends=backends,
                notifier=notifier,
                policy=effective_tuning.worker_policy,
                retry_delay=effective_tuning.retry_delay,
            )
            await workers.start()
            stack.push_async_callback(workers.stop)
            yield cls(
                store=store,
                backends=backends,
                service=service,
                workers=workers,
                event_polling_policy=effective_tuning.event_polling_policy,
            )


class RuntimeProvider:
    """Share lifespan runtimes and own fallback runtimes for direct/Docket calls."""

    def __init__(self, tuning: RuntimeTuning | None = None) -> None:
        self._tuning = tuning or RuntimeTuning()
        self._installed: MCPRuntime | None = None
        self._temporary: MCPRuntime | None = None
        self._temporary_context: AbstractAsyncContextManager[MCPRuntime] | None = None
        self._temporary_borrows = 0
        self._lock = asyncio.Lock()

    @property
    def tuning(self) -> RuntimeTuning:
        """Return the tuning used by the next owned runtime."""
        return self._tuning

    @asynccontextmanager
    async def install(self, runtime: MCPRuntime) -> AsyncIterator[None]:
        """Install one externally owned lifespan runtime for tool borrowing."""
        async with self._lock:
            if self._installed is not None or self._temporary is not None:
                raise RuntimeError("a runtime is already installed")
            self._installed = runtime
        try:
            yield
        finally:
            async with self._lock:
                if self._installed is runtime:
                    self._installed = None

    @asynccontextmanager
    async def borrow(self) -> AsyncIterator[MCPRuntime]:
        """Borrow the installed runtime or share one provider-owned fallback."""
        owns_temporary_borrow = False
        async with self._lock:
            if self._installed is not None:
                borrowed = self._installed
            else:
                if self._temporary is None:
                    temporary_context = MCPRuntime.open(self._tuning)
                    temporary = await temporary_context.__aenter__()
                    self._temporary_context = temporary_context
                    self._temporary = temporary
                borrowed = self._temporary
                self._temporary_borrows += 1
                owns_temporary_borrow = True
        try:
            yield borrowed
        finally:
            context_to_close: AbstractAsyncContextManager[MCPRuntime] | None = None
            if owns_temporary_borrow:
                async with self._lock:
                    self._temporary_borrows -= 1
                    if self._temporary_borrows == 0:
                        context_to_close = self._temporary_context
                        self._temporary_context = None
                        self._temporary = None
                if context_to_close is not None:
                    await context_to_close.__aexit__(None, None, None)

    @contextmanager
    def override_tuning(self, tuning: RuntimeTuning) -> Iterator[None]:
        """Temporarily replace timing dependencies before any runtime is active."""
        if self._installed is not None or self._temporary is not None:
            raise RuntimeError("runtime is installed; tuning cannot be overridden")
        previous = self._tuning
        self._tuning = tuning
        try:
            yield
        finally:
            self._tuning = previous


runtime_provider = RuntimeProvider()
