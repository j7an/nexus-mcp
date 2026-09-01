"""Focused contracts for the durable-store persistence codec."""

from datetime import UTC, datetime

import pytest

from nexus_mcp.core import RequestedExecutionConfig, TurnOperation
from nexus_mcp.jobs._store_codec import (
    InvalidCursorError,
    StoreSchemaError,
    _canonical_json,
    _datetime_to_ms,
    _decode_cursor,
    _decode_json_object,
    _decode_model,
    _decode_operation,
    _encode_cursor,
    _ms_to_datetime,
    _normalize_datetime,
)


def test_canonical_json_is_compact_stable_and_unicode_preserving() -> None:
    """Persisted JSON must be deterministic without escaping Unicode text."""
    assert _canonical_json({"z": 2, "message": "héllo", "a": 1}) == (
        '{"a":1,"message":"héllo","z":2}'
    )


def test_datetime_codec_round_trips_the_persisted_millisecond_precision() -> None:
    """Timestamp hydration must reconstruct the exact stored UTC millisecond."""
    timestamp = datetime(2026, 8, 30, 20, 1, 2, 345000, tzinfo=UTC)

    assert _ms_to_datetime(_datetime_to_ms(timestamp)) == timestamp


def test_datetime_codec_rejects_a_naive_timestamp() -> None:
    """Persisted timestamps must retain an explicit timezone."""
    with pytest.raises(ValueError, match="timezone-aware UTC datetime"):
        _normalize_datetime(datetime(2026, 8, 30, 20, 1, 2))


def test_cursor_codec_round_trips_the_exact_keyset_position() -> None:
    """A generated cursor must preserve both ordering components."""
    encoded = _encode_cursor(1_777_777_777_000, "job-abc")

    assert _decode_cursor(encoded) == (1_777_777_777_000, "job-abc")


@pytest.mark.parametrize(
    "cursor",
    [
        "malformed!",
        "e30",
        "eyJ2IjoyfQ",
        "eyJjcmVhdGVkX2F0X21zIjoiYmFkIiwiam9iX2lkIjoiam9iLTEiLCJ2IjoxfQ",
        "eyJjcmVhdGVkX2F0X21zIjoxLCJqb2JfaWQiOiIiLCJ2IjoxfQ",
    ],
)
def test_cursor_codec_rejects_malformed_or_unsupported_payloads(cursor: str) -> None:
    """Malformed and future cursor payloads must retain the public error contract."""
    with pytest.raises(InvalidCursorError, match="invalid stored-job cursor"):
        _decode_cursor(cursor)


def test_json_object_decoder_rejects_an_unknown_schema_version() -> None:
    """Typed JSON cannot be decoded under an unsupported schema version."""
    with pytest.raises(StoreSchemaError, match="unsupported payload_schema_version: 2"):
        _decode_json_object('{"status":"queued"}', 2, "payload_schema_version")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("not-json", "payload_schema_version payload is invalid JSON"),
        ("[]", "payload_schema_version payload is not an object"),
    ],
)
def test_json_object_decoder_rejects_malformed_payloads(payload: str, message: str) -> None:
    """Persisted object columns must contain valid JSON objects."""
    with pytest.raises(StoreSchemaError, match=message):
        _decode_json_object(payload, 1, "payload_schema_version")


def test_operation_decoder_rejects_an_invalid_typed_payload() -> None:
    """Operation JSON must validate against the closed operation union."""
    with pytest.raises(StoreSchemaError, match="operation_json is not a valid typed operation"):
        _decode_operation('{"kind":"turn"}', 1, "turn")


def test_operation_decoder_rejects_a_kind_column_mismatch() -> None:
    """The indexed operation kind must agree with the typed payload."""
    payload = TurnOperation(prompt="Inspect the workspace").model_dump_json()

    with pytest.raises(StoreSchemaError, match="operation_kind does not match operation_json"):
        _decode_operation(payload, 1, "review")


def test_model_decoder_rejects_an_invalid_typed_payload() -> None:
    """Typed model columns must reject values outside their persisted schema."""
    with pytest.raises(
        StoreSchemaError, match="requested_config_schema_version payload is invalid"
    ):
        _decode_model(
            "[]",
            1,
            "requested_config_schema_version",
            RequestedExecutionConfig,
        )
