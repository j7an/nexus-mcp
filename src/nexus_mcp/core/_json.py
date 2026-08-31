"""Bounded immutable JSON helpers for framework-independent core contracts."""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from pydantic import JsonValue

MAX_JSON_BYTES = 1_048_576
MAX_JSON_DEPTH = 32
MAX_JSON_CONTAINER_ITEMS = 4096


def _validate_json_structure(value: object, *, depth: int = 0) -> None:
    if isinstance(value, Mapping):
        container_depth = depth + 1
        if container_depth > MAX_JSON_DEPTH:
            raise ValueError(f"JSON value exceeds maximum nesting depth of {MAX_JSON_DEPTH}")
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ValueError(
                f"JSON mapping exceeds maximum item count of {MAX_JSON_CONTAINER_ITEMS}"
            )
        for item in value.values():
            _validate_json_structure(item, depth=container_depth)
        return
    if isinstance(value, (list, tuple)):
        container_depth = depth + 1
        if container_depth > MAX_JSON_DEPTH:
            raise ValueError(f"JSON value exceeds maximum nesting depth of {MAX_JSON_DEPTH}")
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ValueError(
                f"JSON sequence exceeds maximum item count of {MAX_JSON_CONTAINER_ITEMS}"
            )
        for item in value:
            _validate_json_structure(item, depth=container_depth)


def thaw_json_value(value: JsonValue) -> JsonValue:
    """Restore ordinary JSON containers at serialization boundaries."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(cast("JsonValue", item)) for item in value]
    return value


def thaw_json_mapping(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Restore an immutable JSON mapping to a plain dictionary."""
    return {key: thaw_json_value(item) for key, item in value.items()}


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        frozen = MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})
        return cast("JsonValue", frozen)
    if isinstance(value, list):
        return cast("JsonValue", tuple(_freeze_json_value(item) for item in value))
    return value


def freeze_bounded_json_value(
    value: JsonValue,
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> JsonValue:
    """Validate exact recursive JSON limits, then return an immutable copy."""
    _validate_json_structure(value)
    plain = thaw_json_value(value)
    encoded = json.dumps(
        plain,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"JSON value exceeds {max_bytes} canonical UTF-8 bytes")
    return _freeze_json_value(value)


def freeze_bounded_json_mapping(
    value: Mapping[str, JsonValue],
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> Mapping[str, JsonValue]:
    """Validate exact recursive JSON limits, then return an immutable mapping copy."""
    frozen = freeze_bounded_json_value(cast("JsonValue", value), max_bytes=max_bytes)
    return cast("Mapping[str, JsonValue]", frozen)
