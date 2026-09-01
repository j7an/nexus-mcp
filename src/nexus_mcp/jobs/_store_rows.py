"""Pure row hydration for the durable SQLite store."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from nexus_mcp.core import (
    TERMINAL_STATES,
    AgentJob,
    AgentOperation,
    AgentSession,
    InputRequest,
    InputResponse,
    JobAttempt,
    JobError,
    JobEvent,
    JobPhase,
    JobResultEnvelope,
    PendingInput,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    Workspace,
)
from nexus_mcp.jobs._store_codec import (
    _INPUT_REQUEST_ADAPTER,
    _INPUT_RESPONSE_ADAPTER,
    _OPERATION_RESULT_ADAPTER,
    StoreSchemaError,
    _decode_json_object,
    _decode_model,
    _decode_operation,
    _ms_to_datetime,
    _optional_ms,
    _require_json_version,
)

type _DecodedJobRow = tuple[
    AgentOperation,
    RequestedExecutionConfig,
    ResolvedExecutionConfig | None,
]
type _DecodedPendingInputRow = tuple[InputRequest, InputResponse | None]


def _workspace_from_row(row: Mapping[str, Any]) -> Workspace:
    return Workspace(
        workspace_id=row["workspace_id"],
        canonical_path=Path(row["canonical_path"]),
        display_name=row["display_name"],
        config_reference=row["config_ref"],
        created_at=_ms_to_datetime(row["created_at_ms"]),
        updated_at=_ms_to_datetime(row["updated_at_ms"]),
    )


def _session_from_row(
    row: Mapping[str, Any],
    provider_references: tuple[ProviderReference, ...],
) -> AgentSession:
    return AgentSession(
        session_id=row["session_id"],
        workspace_id=row["workspace_id"],
        backend_id=row["backend_id"],
        owner_id=row["owner_id"],
        access_policy=row["access_policy"],
        parent_session_id=row["parent_session_id"],
        provider_references=provider_references,
        created_at=_ms_to_datetime(row["created_at_ms"]),
        updated_at=_ms_to_datetime(row["updated_at_ms"]),
    )


def _job_from_row(
    row: Mapping[str, Any],
    source_checkpoint: tuple[ProviderReference, ...],
    *,
    _decoded: _DecodedJobRow | None = None,
) -> AgentJob:
    operation, requested_config, resolved_config = (
        _decode_job_row(row) if _decoded is None else _decoded
    )
    return AgentJob(
        job_id=row["job_id"],
        workspace_id=row["workspace_id"],
        backend_id=row["backend_id"],
        owner_id=row["owner_id"],
        operation=operation,
        requested_config=requested_config,
        request_hash=row["request_hash"],
        access_policy=row["access_policy"],
        session_id=row["session_id"],
        idempotency_key=row["idempotency_key"],
        source_checkpoint=source_checkpoint,
        state=row["state"],
        resolved_config=resolved_config,
        cancel_requested_at=_optional_ms(row["cancel_requested_at_ms"]),
        lease_owner_id=row["lease_owner"],
        lease_generation=row["lease_generation"] or None,
        lease_expires_at=_optional_ms(row["lease_expires_at_ms"]),
        retry_at=_optional_ms(row["retry_at_ms"]),
        created_at=_ms_to_datetime(row["created_at_ms"]),
        updated_at=_ms_to_datetime(row["updated_at_ms"]),
        completed_at=_optional_ms(row["terminal_at_ms"]),
    )


def _decode_job_row(row: Mapping[str, Any]) -> _DecodedJobRow:
    operation = _decode_operation(
        row["operation_json"],
        row["operation_schema_version"],
        row["operation_kind"],
    )
    requested_config = _decode_model(
        row["requested_config_json"],
        row["requested_config_schema_version"],
        "requested_config_schema_version",
        RequestedExecutionConfig,
    )
    resolved_config = None
    if row["resolved_config_json"] is not None:
        resolved_config = _decode_model(
            row["resolved_config_json"],
            row["resolved_config_schema_version"],
            "resolved_config_schema_version",
            ResolvedExecutionConfig,
        )
    elif row["resolved_config_schema_version"] is not None:
        raise StoreSchemaError("resolved_config_schema_version has no typed payload")
    return operation, requested_config, resolved_config


def _job_result_from_row(
    job_id: str,
    row: Mapping[str, Any],
) -> JobResultEnvelope | JobError | None:
    job_state = row["job_state"]
    has_result = row["result_job_id"] is not None
    if job_state not in TERMINAL_STATES:
        if has_result:
            raise StoreSchemaError("nonterminal job has a terminal result row")
        return None
    if not has_result:
        raise StoreSchemaError("terminal job is missing its result row")

    outcome_kind = row["outcome_kind"]
    payload_json = row["payload_json"]
    payload_version = row["payload_schema_version"]
    error_json = row["error_json"]
    error_version = row["error_schema_version"]
    if outcome_kind == "failed":
        if (
            job_state != "failed"
            or error_json is None
            or error_version is None
            or payload_json is not None
            or payload_version is not None
        ):
            raise StoreSchemaError("failed job result row has an invalid state or shape")
        return _decode_model(
            error_json,
            error_version,
            "error_schema_version",
            JobError,
        )
    if outcome_kind == "succeeded":
        if (
            job_state != "completed"
            or payload_json is None
            or payload_version is None
            or error_json is not None
            or error_version is not None
        ):
            raise StoreSchemaError("succeeded job result row has an invalid state or shape")
        _require_json_version(payload_version, "payload_schema_version")
        try:
            payload = _OPERATION_RESULT_ADAPTER.validate_json(payload_json)
        except (ValidationError, ValueError) as error:
            raise StoreSchemaError("job result payload is invalid") from error
        return JobResultEnvelope(
            job_id=job_id,
            payload=payload,
            completed_at=_ms_to_datetime(row["created_at_ms"]),
        )
    if outcome_kind == "cancelled":
        if (
            job_state != "cancelled"
            or payload_json is not None
            or payload_version is not None
            or error_json is not None
            or error_version is not None
        ):
            raise StoreSchemaError("cancelled job result row has an invalid state or shape")
        return None
    raise StoreSchemaError(f"unknown job result outcome: {outcome_kind}")


def _job_attempt_from_row(row: Mapping[str, Any]) -> JobAttempt:
    error = None
    if row["error_json"] is not None:
        error = _decode_model(
            row["error_json"],
            row["error_schema_version"],
            "error_schema_version",
            JobError,
        )
    elif row["error_schema_version"] is not None:
        raise StoreSchemaError("attempt error schema version has no payload")
    return JobAttempt(
        job_id=row["job_id"],
        attempt_number=row["attempt_number"],
        phase=cast("JobPhase", row["phase"]),
        worker_id=row["owner_id"],
        lease_generation=row["lease_generation"],
        lease_expires_at=_optional_ms(row["lease_expires_at_ms"]),
        heartbeat_at=_optional_ms(row["heartbeat_at_ms"]),
        retry_classification=row["retry_classification"],
        reconciliation_classification=row["reconciliation_classification"],
        started_at=_ms_to_datetime(row["started_at_ms"]),
        ended_at=_optional_ms(row["ended_at_ms"]),
        error_code=None if error is None else error.code,
        error_message=None if error is None else error.message,
    )


def _pending_input_from_row(
    row: Mapping[str, Any],
    provider_reference: ProviderReference | None,
    *,
    _decoded: _DecodedPendingInputRow | None = None,
) -> PendingInput:
    request, response = _decode_pending_input_row(row) if _decoded is None else _decoded
    try:
        return PendingInput(
            input_id=row["input_id"],
            job_id=row["job_id"],
            request=request,
            provider_reference=provider_reference,
            created_at=_ms_to_datetime(row["created_at_ms"]),
            resolved_at=_optional_ms(row["resolved_at_ms"]),
            response=response,
        )
    except ValidationError as error:
        raise StoreSchemaError("pending input row is invalid") from error


def _decode_pending_input_row(row: Mapping[str, Any]) -> _DecodedPendingInputRow:
    _require_json_version(row["request_schema_version"], "request_schema_version")
    try:
        request = _INPUT_REQUEST_ADAPTER.validate_json(row["request_json"])
    except (ValidationError, ValueError) as error:
        raise StoreSchemaError("pending input request is invalid") from error
    if request.kind != row["kind"]:
        raise StoreSchemaError("pending input kind does not match its request")
    response = None
    if row["response_json"] is not None:
        _require_json_version(row["response_schema_version"], "response_schema_version")
        try:
            response = _INPUT_RESPONSE_ADAPTER.validate_json(row["response_json"])
        except (ValidationError, ValueError) as error:
            raise StoreSchemaError("pending input response is invalid") from error
    elif row["response_schema_version"] is not None:
        raise StoreSchemaError("pending input response schema version has no payload")
    if (row["status"] == "pending") != (response is None):
        raise StoreSchemaError("pending input status does not match its response")
    return request, response


def _event_from_row(
    row: Mapping[str, Any],
    provider_reference: ProviderReference | None,
    *,
    _payload: dict[str, Any] | None = None,
) -> JobEvent:
    payload = _decode_event_payload(row) if _payload is None else _payload
    try:
        return JobEvent(
            job_id=row["job_id"],
            sequence=row["sequence"],
            type=row["event_type"],
            payload=payload,
            payload_schema_version=row["payload_schema_version"],
            occurred_at=_ms_to_datetime(row["created_at_ms"]),
            attempt_number=row["attempt_number"],
            provider_event_type=row["provider_event_type"],
            provider_reference=provider_reference,
        )
    except ValidationError as error:
        raise StoreSchemaError("job event row is invalid") from error


def _decode_event_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return _decode_json_object(
        row["payload_json"],
        row["payload_schema_version"],
        "payload_schema_version",
    )


def _provider_references_from_pairs(
    pairs: Iterable[tuple[str, str]],
) -> tuple[ProviderReference, ...]:
    references: list[ProviderReference] = []
    seen: set[tuple[str, str]] = set()
    for kind, value in pairs:
        identity_pair = (kind, value)
        if identity_pair in seen:
            continue
        seen.add(identity_pair)
        references.append(ProviderReference(kind=kind, value=value))
    return tuple(references)
