"""Typed FastMCP tools over the durable framework-independent job service."""

from collections.abc import Awaitable, Callable, Mapping

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nexus_mcp.core import (
    AccessContext,
    AccessPolicy,
    BackendStatus,
    CancelReceipt,
    ExecutionConfigValues,
    ForkOperation,
    InputResolutionReceipt,
    InputResponse,
    JobHandle,
    JobListPage,
    JobResultResponse,
    JobState,
    JobStatus,
    NexusCoreError,
    ReviewDelivery,
    ReviewOperation,
    ReviewTarget,
    TurnOperation,
    WorkspaceSelector,
)
from nexus_mcp.jobs import AgentJobService
from nexus_mcp.mcp.access import local_access_context
from nexus_mcp.mcp.runtime import runtime_provider

__all__ = [
    "BackendListResult",
    "agent_backends",
    "agent_cancel",
    "agent_continue",
    "agent_diagnose",
    "agent_fork",
    "agent_list",
    "agent_respond",
    "agent_result",
    "agent_review",
    "agent_start",
    "agent_status",
    "register_job_tools",
]


class BackendListResult(BaseModel):
    """Stable typed wrapper for backend status discovery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[BackendStatus, ...] = Field(default=(), max_length=256)


class _AgentToolFailure(BaseModel):
    """Machine-readable domain error serialized through MCP's error channel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4096)


async def _invoke[ResultT](
    operation: Callable[[AgentJobService, AccessContext], Awaitable[ResultT]],
) -> ResultT:
    try:
        async with runtime_provider.borrow() as runtime:
            return await operation(runtime.service, local_access_context())
    except NexusCoreError as error:
        failure = _AgentToolFailure(code=error.code, message=str(error))
        raise ToolError(failure.model_dump_json()) from error


async def agent_start(
    *,
    workspace: WorkspaceSelector,
    backend: str,
    prompt: str,
    context: Mapping[str, JsonValue] | None = None,
    file_refs: list[str] | None = None,
    config: ExecutionConfigValues | None = None,
    access_policy: AccessPolicy = "private",
    idempotency_key: str | None = None,
) -> JobHandle:
    """Start a durable session turn in an explicitly selected workspace."""
    operation = TurnOperation(
        prompt=prompt,
        context={} if context is None else context,
        file_refs=() if file_refs is None else tuple(file_refs),
    )
    explicit_config = config or ExecutionConfigValues()
    return await _invoke(
        lambda service, access: service.start(
            workspace=workspace,
            access=access,
            backend_id=backend,
            operation=operation,
            explicit_config=explicit_config,
            access_policy=access_policy,
            idempotency_key=idempotency_key,
        )
    )


async def agent_continue(
    *,
    workspace: WorkspaceSelector,
    session_id: str,
    prompt: str,
    context: Mapping[str, JsonValue] | None = None,
    file_refs: list[str] | None = None,
    config: ExecutionConfigValues | None = None,
    idempotency_key: str | None = None,
) -> JobHandle:
    """Queue a durable turn against an existing authorized session."""
    operation = TurnOperation(
        prompt=prompt,
        context={} if context is None else context,
        file_refs=() if file_refs is None else tuple(file_refs),
    )
    explicit_config = config or ExecutionConfigValues()
    return await _invoke(
        lambda service, access: service.continue_session(
            workspace=workspace,
            access=access,
            session_id=session_id,
            operation=operation,
            explicit_config=explicit_config,
            idempotency_key=idempotency_key,
        )
    )


async def agent_fork(
    *,
    workspace: WorkspaceSelector,
    session_id: str,
    prompt: str | None = None,
    context: Mapping[str, JsonValue] | None = None,
    file_refs: list[str] | None = None,
    config: ExecutionConfigValues | None = None,
    idempotency_key: str | None = None,
) -> JobHandle:
    """Fork an existing session when its backend advertises that capability."""
    operation = ForkOperation(
        prompt=prompt,
        context={} if context is None else context,
        file_refs=() if file_refs is None else tuple(file_refs),
    )
    explicit_config = config or ExecutionConfigValues()
    return await _invoke(
        lambda service, access: service.fork_session(
            workspace=workspace,
            access=access,
            session_id=session_id,
            operation=operation,
            explicit_config=explicit_config,
            idempotency_key=idempotency_key,
        )
    )


async def agent_review(
    *,
    workspace: WorkspaceSelector,
    session_id: str,
    target: ReviewTarget,
    delivery: ReviewDelivery = "inline",
    instructions: str | None = None,
    context: Mapping[str, JsonValue] | None = None,
    file_refs: list[str] | None = None,
    config: ExecutionConfigValues | None = None,
    idempotency_key: str | None = None,
) -> JobHandle:
    """Queue a typed review when the session backend supports its target and delivery."""
    operation = ReviewOperation(
        target=target,
        delivery=delivery,
        instructions=instructions,
        context={} if context is None else context,
        file_refs=() if file_refs is None else tuple(file_refs),
    )
    explicit_config = config or ExecutionConfigValues()
    return await _invoke(
        lambda service, access: service.review(
            workspace=workspace,
            access=access,
            session_id=session_id,
            operation=operation,
            explicit_config=explicit_config,
            idempotency_key=idempotency_key,
        )
    )


async def agent_diagnose(
    *,
    workspace: WorkspaceSelector,
    backend: str,
    config: ExecutionConfigValues | None = None,
    idempotency_key: str | None = None,
) -> JobHandle:
    """Queue backend diagnostics without conditionally hiding the tool."""
    explicit_config = config or ExecutionConfigValues()
    return await _invoke(
        lambda service, access: service.diagnose(
            workspace=workspace,
            access=access,
            backend_id=backend,
            explicit_config=explicit_config,
            idempotency_key=idempotency_key,
        )
    )


async def agent_status(*, workspace: WorkspaceSelector, job_id: str) -> JobStatus:
    """Return one authorized durable job status projection."""
    return await _invoke(
        lambda service, access: service.status(
            workspace=workspace,
            access=access,
            job_id=job_id,
        )
    )


async def agent_result(*, workspace: WorkspaceSelector, job_id: str) -> JobResultResponse:
    """Return the pending, succeeded, failed, or cancelled durable result variant."""
    return await _invoke(
        lambda service, access: service.result(
            workspace=workspace,
            access=access,
            job_id=job_id,
        )
    )


async def agent_cancel(*, workspace: WorkspaceSelector, job_id: str) -> CancelReceipt:
    """Request idempotent cancellation of one authorized durable job."""
    return await _invoke(
        lambda service, access: service.cancel(
            workspace=workspace,
            access=access,
            job_id=job_id,
        )
    )


async def agent_respond(
    *,
    workspace: WorkspaceSelector,
    job_id: str,
    input_id: str,
    response: InputResponse,
) -> InputResolutionReceipt:
    """Resolve one pending typed provider interaction for a durable job."""
    return await _invoke(
        lambda service, access: service.respond(
            workspace=workspace,
            access=access,
            job_id=job_id,
            input_id=input_id,
            response=response,
        )
    )


async def agent_list(
    *,
    workspace: WorkspaceSelector,
    states: list[JobState] | None = None,
    limit: int = 50,
    cursor: str | None = None,
) -> JobListPage:
    """List one authorized page of durable jobs for an explicit workspace."""
    selected_states = frozenset() if states is None else frozenset(states)
    return await _invoke(
        lambda service, access: service.list_jobs(
            workspace=workspace,
            access=access,
            states=selected_states,
            limit=limit,
            cursor=cursor,
        )
    )


async def agent_backends(*, workspace: WorkspaceSelector) -> BackendListResult:
    """Return deterministic descriptors with fresh availability for a workspace."""
    items = await _invoke(
        lambda service, access: service.list_backends(workspace=workspace, access=access)
    )
    return BackendListResult(items=items)


def _annotations(
    title: str,
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool,
    open_world: bool,
) -> ToolAnnotations:
    return ToolAnnotations(
        title=title,
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
    )


def register_job_tools(server: FastMCP) -> None:
    """Register the complete durable-job surface independent of backend health."""
    execution_tools = (
        (agent_start, "Start Agent Job"),
        (agent_continue, "Continue Agent Session"),
        (agent_fork, "Fork Agent Session"),
        (agent_review, "Review With Agent"),
        (agent_diagnose, "Diagnose Agent Backend"),
    )
    for execution_function, title in execution_tools:
        server.tool(
            annotations=_annotations(
                title,
                read_only=False,
                destructive=True,
                idempotent=False,
                open_world=True,
            ),
            tags={"agent-jobs"},
        )(execution_function)

    observation_tools = (
        (agent_status, "Get Agent Job Status"),
        (agent_result, "Get Agent Job Result"),
        (agent_list, "List Agent Jobs"),
        (agent_backends, "List Agent Backends"),
    )
    for observation_function, title in observation_tools:
        server.tool(
            annotations=_annotations(
                title,
                read_only=True,
                destructive=False,
                idempotent=True,
                open_world=False,
            ),
            tags={"agent-jobs"},
        )(observation_function)

    server.tool(
        annotations=_annotations(
            "Cancel Agent Job",
            read_only=False,
            destructive=True,
            idempotent=True,
            open_world=False,
        ),
        tags={"agent-jobs"},
    )(agent_cancel)
    server.tool(
        annotations=_annotations(
            "Respond to Agent Input",
            read_only=False,
            destructive=False,
            idempotent=True,
            open_world=False,
        ),
        tags={"agent-jobs"},
    )(agent_respond)
