"""Runtime-owned registry for typed backend implementations."""

from collections.abc import Iterable

from nexus_mcp.backends.base import AgentBackend
from nexus_mcp.core import (
    BackendDisabledError,
    BackendStatus,
    BackendUnknownError,
    OperationKind,
    UnsupportedCapabilityError,
    Workspace,
)

__all__ = ["BackendManager"]


class BackendManager:
    """Register backend runtimes once and expose deterministic capability discovery."""

    def __init__(self, backends: Iterable[AgentBackend]) -> None:
        self._backends: dict[str, AgentBackend] = {}
        self._closed_backend_ids: set[str] = set()
        for backend in backends:
            backend_id = backend.descriptor.backend_id
            if backend_id in self._backends:
                raise ValueError(f"Duplicate backend id: {backend_id}")
            self._backends[backend_id] = backend

    def get(self, backend_id: str) -> AgentBackend:
        """Return one registered backend or raise the stable unknown-backend error."""
        try:
            return self._backends[backend_id]
        except KeyError as exc:
            raise BackendUnknownError(backend_id) from exc

    def require_operation(self, backend_id: str, operation: OperationKind) -> AgentBackend:
        """Return a registered backend only when it is enabled and advertises the operation."""
        backend = self.get(backend_id)
        if not backend.descriptor.enabled:
            raise BackendDisabledError(backend_id)
        if operation not in backend.descriptor.capabilities.operations:
            raise UnsupportedCapabilityError(backend_id, operation)
        return backend

    async def list_statuses(self, workspace: Workspace) -> tuple[BackendStatus, ...]:
        """Return registered backends in stable identifier order with fresh health observations."""
        statuses: list[BackendStatus] = []
        for backend in sorted(self._backends.values(), key=lambda item: item.descriptor.backend_id):
            statuses.append(
                BackendStatus(
                    descriptor=backend.descriptor,
                    availability=await backend.check_availability(workspace),
                )
            )
        return tuple(statuses)

    async def close(self) -> None:
        """Close every registered runtime no more than once, in deterministic order."""
        for backend_id, backend in sorted(self._backends.items()):
            if backend_id in self._closed_backend_ids:
                continue
            self._closed_backend_ids.add(backend_id)
            await backend.close()
