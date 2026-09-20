"""Closed, framework-independent operation contracts."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from nexus_mcp.core._json import freeze_bounded_json_mapping, thaw_json_mapping

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
        """Bound and protect admitted context from mutation after validation."""
        return freeze_bounded_json_mapping(value)

    @field_serializer("context")
    def serialize_context(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Restore ordinary JSON containers at serialization boundaries."""
        return thaw_json_mapping(value)


class TurnOperation(_ContextOperation):
    """One prompt executed in a new or existing Nexus session."""

    kind: Literal["turn"] = "turn"
    prompt: str = Field(min_length=1, max_length=131_072)
    output_schema: Mapping[str, JsonValue] | None = Field(
        default=None, max_length=_MAX_CONTEXT_ITEMS
    )

    @field_validator("output_schema", mode="after")
    @classmethod
    def freeze_output_schema(
        cls, value: Mapping[str, JsonValue] | None
    ) -> Mapping[str, JsonValue] | None:
        """Bound and protect a caller-supplied JSON Schema from mutation."""
        return None if value is None else freeze_bounded_json_mapping(value)

    @field_serializer("output_schema")
    def serialize_output_schema(
        self, value: Mapping[str, JsonValue] | None
    ) -> dict[str, JsonValue] | None:
        """Restore ordinary JSON containers before model-level omission handling."""
        return None if value is None else thaw_json_mapping(value)

    @model_serializer(mode="wrap")
    def omit_absent_output_schema(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep serialized turns byte-identical to pre-schema releases when no schema is set."""
        data: dict[str, Any] = handler(self)
        if self.output_schema is None:
            data.pop("output_schema", None)
        else:
            data["output_schema"] = thaw_json_mapping(self.output_schema)
        return data


class ForkOperation(_ContextOperation):
    """A new child session created from an admitted source checkpoint."""

    kind: Literal["fork"] = "fork"
    prompt: str | None = Field(default=None, min_length=1, max_length=131_072)


class ReviewTarget(_OperationModel):
    """A bounded, provider-neutral target for a code review."""

    kind: ReviewTargetKind
    reference: str | None = Field(default=None, min_length=1, max_length=4096)

    @model_validator(mode="after")
    def require_named_target_reference(self) -> "ReviewTarget":
        """A branch or commit review must identify the target it will inspect."""
        if self.kind in ("branch", "commit") and self.reference is None:
            raise ValueError("branch and commit review targets require a reference")
        return self


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
