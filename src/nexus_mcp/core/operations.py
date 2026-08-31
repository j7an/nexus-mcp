"""Closed, framework-independent operation contracts."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
)

__all__ = [
    "AgentOperation",
    "DiagnosticsOperation",
    "ForkOperation",
    "OperationKind",
    "ReviewDelivery",
    "ReviewOperation",
    "ReviewTarget",
    "ReviewTargetKind",
    "TurnOperation",
]

type OperationKind = Literal["turn", "fork", "review", "diagnostics"]
type ReviewDelivery = Literal["inline", "detached"]
type ReviewTargetKind = Literal["working_tree", "branch", "commit", "pull_request"]

_MAX_CONTEXT_ITEMS = 256
_MAX_FILE_REFS = 256


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        frozen = MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
        return cast("JsonValue", frozen)
    if isinstance(value, list):
        return cast("JsonValue", tuple(_freeze_json_value(item) for item in value))
    return value


def _freeze_json_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})


def _thaw_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(cast("JsonValue", item)) for item in value]
    return value


class _OperationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _ContextOperation(_OperationModel):
    context: Mapping[str, JsonValue] = Field(
        default_factory=dict,
        max_length=_MAX_CONTEXT_ITEMS,
        validate_default=True,
    )
    file_refs: tuple[Annotated[str, Field(min_length=1, max_length=4096)], ...] = Field(
        default=(),
        max_length=_MAX_FILE_REFS,
    )

    @field_validator("context", mode="after")
    @classmethod
    def freeze_context(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Protect admitted context from mutation after validation."""
        return _freeze_json_mapping(value)

    @field_serializer("context")
    def serialize_context(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Restore ordinary JSON containers at serialization boundaries."""
        return {key: _thaw_json_value(item) for key, item in value.items()}


class TurnOperation(_ContextOperation):
    """One prompt executed in a new or existing Nexus session."""

    kind: Literal["turn"] = "turn"
    prompt: str = Field(min_length=1, max_length=131_072)


class ForkOperation(_ContextOperation):
    """A new child session created from an admitted source checkpoint."""

    kind: Literal["fork"] = "fork"
    prompt: str | None = Field(default=None, min_length=1, max_length=131_072)


class ReviewTarget(_OperationModel):
    """A bounded, provider-neutral target for a code review."""

    kind: ReviewTargetKind
    reference: str | None = Field(default=None, min_length=1, max_length=4096)


class ReviewOperation(_ContextOperation):
    """A normalized review request with explicit delivery semantics."""

    kind: Literal["review"] = "review"
    target: ReviewTarget
    delivery: ReviewDelivery = "inline"
    instructions: str | None = Field(default=None, min_length=1, max_length=131_072)


class DiagnosticsOperation(_OperationModel):
    """A sessionless request for current backend diagnostics."""

    kind: Literal["diagnostics"] = "diagnostics"


type AgentOperation = Annotated[
    TurnOperation | ForkOperation | ReviewOperation | DiagnosticsOperation,
    Field(discriminator="kind"),
]
