"""Pure persistence codecs shared by the durable SQLite adapter."""

import base64
import binascii
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError

from nexus_mcp.core import (
    AgentOperation,
    InputRequest,
    InputResponse,
    OperationResult,
)

_JSON_SCHEMA_VERSION = 1
_CURSOR_VERSION = 1
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_OPERATION_ADAPTER: TypeAdapter[AgentOperation] = TypeAdapter(AgentOperation)
_INPUT_REQUEST_ADAPTER: TypeAdapter[InputRequest] = TypeAdapter(InputRequest)
_INPUT_RESPONSE_ADAPTER: TypeAdapter[InputResponse] = TypeAdapter(InputResponse)
_OPERATION_RESULT_ADAPTER: TypeAdapter[OperationResult] = TypeAdapter(OperationResult)


class StoreSchemaError(RuntimeError):
    """Raised when persisted typed JSON uses an unsupported schema version."""


class InvalidCursorError(ValueError):
    """Raised when a stored-job keyset cursor is malformed or unsupported."""


def _canonical_workspace_path(path: Path) -> str:
    return os.path.normcase(str(path))


def _datetime_to_ms(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    delta = normalized - _EPOCH
    return delta.days * 86_400_000 + delta.seconds * 1_000 + delta.microseconds // 1_000


def _ms_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1_000, UTC)


def _optional_ms(value: int | None) -> datetime | None:
    return None if value is None else _ms_to_datetime(value)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must be a timezone-aware UTC datetime")
    return value.astimezone(UTC)


def _canonical_json(value: BaseModel | object) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_json_object(payload: str, version: int, version_column: str) -> dict[str, Any]:
    _require_json_version(version, version_column)
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, TypeError) as error:
        raise StoreSchemaError(f"{version_column} payload is invalid JSON") from error
    if not isinstance(decoded, dict):
        raise StoreSchemaError(f"{version_column} payload is not an object")
    return decoded


def _require_json_version(version: int | None, column: str) -> None:
    if version != _JSON_SCHEMA_VERSION:
        raise StoreSchemaError(f"unsupported {column}: {version}")


def _decode_operation(payload: str, version: int, expected_kind: str) -> AgentOperation:
    _require_json_version(version, "operation_schema_version")
    try:
        operation = _OPERATION_ADAPTER.validate_json(payload)
    except (ValidationError, ValueError) as error:
        raise StoreSchemaError("operation_json is not a valid typed operation") from error
    if operation.kind != expected_kind:
        raise StoreSchemaError("operation_kind does not match operation_json")
    return operation


def _decode_model[ModelT: BaseModel](
    payload: str,
    version: int | None,
    version_column: str,
    model: type[ModelT],
) -> ModelT:
    _require_json_version(version, version_column)
    try:
        return model.model_validate_json(payload)
    except (ValidationError, ValueError) as error:
        raise StoreSchemaError(f"{version_column} payload is invalid") from error


def _encode_cursor(created_at_ms: int, job_id: str) -> str:
    payload = _canonical_json(
        {"v": _CURSOR_VERSION, "created_at_ms": created_at_ms, "job_id": job_id}
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> tuple[int, str]:
    if re.fullmatch(r"[A-Za-z0-9_-]+", cursor) is None:
        raise InvalidCursorError("invalid stored-job cursor")
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.b64decode(
            (cursor + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"v", "created_at_ms", "job_id"}:
            raise ValueError
        if type(payload["v"]) is not int or payload["v"] != _CURSOR_VERSION:
            raise ValueError
        if type(payload["created_at_ms"]) is not int:
            raise ValueError
        if not isinstance(payload["job_id"], str) or not payload["job_id"]:
            raise ValueError
    except (
        binascii.Error,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        UnicodeEncodeError,
        ValueError,
    ) as error:
        raise InvalidCursorError("invalid stored-job cursor") from error
    return payload["created_at_ms"], payload["job_id"]
