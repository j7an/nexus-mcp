"""Normalized provider-neutral input request and response contracts."""

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from nexus_mcp.core.models import ProviderReference

__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalResponse",
    "FormField",
    "FormRequest",
    "FormResponse",
    "InputRequest",
    "InputResolutionReceipt",
    "InputResponse",
    "PendingInput",
    "PermissionRequest",
    "PermissionResponse",
    "QuestionRequest",
    "QuestionResponse",
]

type ApprovalDecision = Literal["approve", "deny"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be a timezone-aware UTC datetime")
    return value.astimezone(UTC)


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        frozen = MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
        return cast("JsonValue", frozen)
    if isinstance(value, list):
        return cast("JsonValue", tuple(_freeze_json_value(item) for item in value))
    return value


def _thaw_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(cast("JsonValue", item)) for item in value]
    return value


class _InteractionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ApprovalRequest(_InteractionModel):
    """A bounded yes/no approval request."""

    kind: Literal["approval"] = "approval"
    prompt: str = Field(min_length=1, max_length=8192)
    risk: str | None = Field(default=None, min_length=1, max_length=4096)
    allowed_decisions: frozenset[ApprovalDecision] = Field(
        default=frozenset({"approve", "deny"}),
        min_length=1,
    )


class PermissionRequest(_InteractionModel):
    """A request for one or more normalized permission scopes."""

    kind: Literal["permission"] = "permission"
    prompt: str = Field(min_length=1, max_length=8192)
    risk: str | None = Field(default=None, min_length=1, max_length=4096)
    requested: frozenset[Annotated[str, Field(min_length=1, max_length=2048)]] = Field(
        min_length=1,
        max_length=256,
    )


class QuestionRequest(_InteractionModel):
    """A normalized free-text or multiple-choice question."""

    kind: Literal["question"] = "question"
    prompt: str = Field(min_length=1, max_length=8192)
    choices: tuple[Annotated[str, Field(min_length=1, max_length=2048)], ...] = Field(
        default=(),
        max_length=256,
    )
    allow_free_text: bool = True


class FormField(_InteractionModel):
    """One bounded text field requested by a provider-neutral form."""

    name: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=4096)
    required: bool = True


class FormRequest(_InteractionModel):
    """A normalized collection of named text fields."""

    kind: Literal["form"] = "form"
    prompt: str = Field(min_length=1, max_length=8192)
    fields: tuple[FormField, ...] = Field(min_length=1, max_length=256)

    @field_validator("fields", mode="after")
    @classmethod
    def require_unique_field_names(cls, value: tuple[FormField, ...]) -> tuple[FormField, ...]:
        """Reject ambiguous response maps."""
        names = [field.name for field in value]
        if len(names) != len(set(names)):
            raise ValueError("form field names must be unique")
        return value


type InputRequest = Annotated[
    ApprovalRequest | PermissionRequest | QuestionRequest | FormRequest,
    Field(discriminator="kind"),
]


class ApprovalResponse(_InteractionModel):
    """A normalized approval decision."""

    kind: Literal["approval"] = "approval"
    decision: ApprovalDecision

    @model_validator(mode="after")
    def validate_allowed_decision(self, info: ValidationInfo) -> "ApprovalResponse":
        """Enforce request-specific approval choices during pending-input validation."""
        allowed = None if info.context is None else info.context.get("allowed_decisions")
        if allowed is not None and self.decision not in allowed:
            raise ValueError("approval decision was not offered by the request")
        return self


class PermissionResponse(_InteractionModel):
    """A normalized subset of requested permission scopes."""

    kind: Literal["permission"] = "permission"
    granted: frozenset[Annotated[str, Field(min_length=1, max_length=2048)]] = Field(
        default=frozenset(),
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_permission_subset(self, info: ValidationInfo) -> "PermissionResponse":
        """Prevent a response from granting authority absent from its request."""
        requested = None if info.context is None else info.context.get("requested")
        if requested is not None and not self.granted.issubset(requested):
            raise ValueError("granted permissions must be a subset of requested permissions")
        return self


class QuestionResponse(_InteractionModel):
    """A normalized answer to one question."""

    kind: Literal["question"] = "question"
    answer: str = Field(min_length=1, max_length=131_072)

    @model_validator(mode="after")
    def validate_answer_shape(self, info: ValidationInfo) -> "QuestionResponse":
        """Require an offered choice when the request disallows free text."""
        if info.context is None:
            return self
        choices = cast("tuple[str, ...]", info.context.get("choices", ()))
        allow_free_text = bool(info.context.get("allow_free_text", True))
        if not allow_free_text and self.answer not in choices:
            raise ValueError("answer must be one of the offered choices")
        return self


class FormResponse(_InteractionModel):
    """A normalized map of form field values."""

    kind: Literal["form"] = "form"
    values: Mapping[str, JsonValue] = Field(max_length=256)

    @field_validator("values", mode="after")
    @classmethod
    def freeze_values(cls, value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Protect resolved form values from mutation."""
        return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})

    @field_serializer("values")
    def serialize_values(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Emit an ordinary JSON object for persistence."""
        return {key: _thaw_json_value(item) for key, item in value.items()}

    @model_validator(mode="after")
    def validate_form_fields(self, info: ValidationInfo) -> "FormResponse":
        """Reject unknown fields and require every mandatory field."""
        if info.context is None:
            return self
        fields = cast("tuple[FormField, ...]", info.context.get("fields", ()))
        allowed = {field.name for field in fields}
        required = {field.name for field in fields if field.required}
        actual = set(self.values)
        if not actual.issubset(allowed):
            raise ValueError("form response contains an unknown field")
        if not required.issubset(actual):
            raise ValueError("form response is missing a required field")
        return self


type InputResponse = Annotated[
    ApprovalResponse | PermissionResponse | QuestionResponse | FormResponse,
    Field(discriminator="kind"),
]


class PendingInput(_InteractionModel):
    """A durable normalized input request awaiting or retaining one response."""

    input_id: str = Field(min_length=1, max_length=256)
    job_id: str = Field(min_length=1, max_length=256)
    request: InputRequest
    provider_reference: ProviderReference | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    resolved_at: datetime | None = None
    response: InputResponse | None = None

    @field_validator("created_at", "resolved_at", mode="after")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        """Store input timestamps in UTC."""
        return None if value is None else _normalize_utc(value)

    def validate_response(self, response: InputResponse) -> InputResponse:
        """Return a provider-neutral response valid for this request's exact shape."""
        values = response.model_dump()
        match self.request:
            case ApprovalRequest(allowed_decisions=allowed_decisions):
                return ApprovalResponse.model_validate(
                    values,
                    context={"allowed_decisions": allowed_decisions},
                )
            case PermissionRequest(requested=requested):
                return PermissionResponse.model_validate(
                    values,
                    context={"requested": requested},
                )
            case QuestionRequest(choices=choices, allow_free_text=allow_free_text):
                return QuestionResponse.model_validate(
                    values,
                    context={"choices": choices, "allow_free_text": allow_free_text},
                )
            case FormRequest(fields=fields):
                return FormResponse.model_validate(values, context={"fields": fields})


class InputResolutionReceipt(_InteractionModel):
    """The durable outcome of an idempotent input response."""

    job_id: str = Field(min_length=1, max_length=256)
    input_id: str = Field(min_length=1, max_length=256)
    status: Literal["resolved"] = "resolved"
    replayed: bool = False
