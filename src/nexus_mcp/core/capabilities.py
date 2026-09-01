"""Static backend capabilities and current availability contracts."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from nexus_mcp.core.models import SandboxMode
from nexus_mcp.core.operations import (
    OperationKind,
    ReviewDelivery,
    ReviewTargetKind,
)

__all__ = [
    "BackendAvailability",
    "BackendCapabilities",
    "BackendDescriptor",
    "BackendStatus",
]


class _CapabilityModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BackendCapabilities(_CapabilityModel):
    """Deterministic feature support advertised before job admission."""

    operations: frozenset[OperationKind]
    cancellation: bool = False
    graceful_interrupt: bool = False
    session_continuation: bool = False
    session_fork: bool = False
    dynamic_models: bool = False
    input_required: bool = False
    structured_output: bool = False
    sandbox_modes: frozenset[SandboxMode] = frozenset()
    review_targets: frozenset[ReviewTargetKind] = frozenset()
    review_deliveries: frozenset[ReviewDelivery] = frozenset()


class BackendDescriptor(_CapabilityModel):
    """Stable backend metadata independent of runtime health."""

    backend_id: str = Field(min_length=1, max_length=256)
    display_name: str = Field(min_length=1, max_length=256)
    capabilities: BackendCapabilities
    enabled: bool = True
    description: str | None = Field(default=None, min_length=1, max_length=2048)


class BackendAvailability(_CapabilityModel):
    """One current, non-authoritative backend diagnostic observation."""

    available: bool
    authenticated: bool | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=4096)
    version: str | None = Field(default=None, min_length=1, max_length=256)
    setup_guidance: str | None = Field(default=None, min_length=1, max_length=4096)
    models: tuple[Annotated[str, Field(min_length=1, max_length=256)], ...] = Field(
        default=(),
        max_length=1024,
    )


class BackendStatus(_CapabilityModel):
    """A deterministic descriptor paired with current availability."""

    descriptor: BackendDescriptor
    availability: BackendAvailability
