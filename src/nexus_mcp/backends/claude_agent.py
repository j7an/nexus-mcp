"""Claude Agent SDK backend: one SDK process per active Nexus turn."""

import asyncio
import re
import shutil
import sys
from collections.abc import Callable
from typing import Any

import claude_agent_sdk
import jsonschema
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    CLINotFoundError,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    fork_session,
)
from claude_agent_sdk._cli_version import __cli_version__
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
from pydantic import ValidationError

from nexus_mcp.backends.base import (
    BackendExecutionContext,
    BackendFailure,
    CancelRequested,
    ReconciliationOutcome,
    RetryDisposition,
    UnknownReconciliationOutcome,
)
from nexus_mcp.backends.claude_policy import WRITE_TOOLS, build_options, decide
from nexus_mcp.config import get_claude_settings_profile, get_runner_defaults
from nexus_mcp.config_resolver import get_agent_env
from nexus_mcp.core import (
    AgentOperation,
    ApprovalPolicy,
    BackendAvailability,
    BackendCapabilities,
    BackendDescriptor,
    BackendEvent,
    ExecutionConfigValues,
    ForkOperation,
    ForkResult,
    JobError,
    JobErrorCode,
    OperationResult,
    PermissionRequest,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    RetryPolicy,
    ReviewOperation,
    ReviewResult,
    SandboxMode,
    TurnOperation,
    TurnResult,
    Workspace,
)

__all__ = ["ClaudeAgentBackend"]

_SESSION = "session"
_EXECUTABLE = re.compile(r"[A-Za-z0-9_./+-]{1,128}")


def _failure(
    code: JobErrorCode,
    message: str,
    disposition: RetryDisposition = "terminal",
) -> BackendFailure:
    """Build a normalized failure from fixed text only; never copy provider text."""
    error = JobError(
        code=code,
        message=message,
        retry_disposition=disposition,
        recoverable=disposition != "terminal",
    )
    return BackendFailure(error, disposition)


def _truncate(message: str, limit: int | None) -> str:
    if limit is None:
        return message
    return message.encode()[:limit].decode(errors="ignore")


def _with_file_refs(prompt: str, file_refs: tuple[str, ...]) -> str:
    if not file_refs:
        return prompt
    return f"{prompt}\n\nFile references:\n" + "\n".join(f"- {path}" for path in file_refs)


def _review_prompt(operation: ReviewOperation) -> str:
    """Return a fixed, read-only prompt for an admitted review operation."""
    target = operation.target
    subject = {
        "working_tree": "the uncommitted changes in the working tree",
        "branch": f"the changes on this branch relative to {target.reference}",
        "commit": f"commit {target.reference}",
    }.get(target.kind, f"{target.kind} {target.reference or ''}".strip())
    parts = [
        (
            f"Review {subject}. Inspect it with the Read, Glob and Grep tools and with git diff, "
            "git log and git show only. Every git command must include both --no-ext-diff and "
            "--no-textconv and must not use quotes, pipes, redirects or --output; any other "
            "command is denied. Do not modify anything. "
            "Report a verdict, a short summary, and concrete findings."
        ),
    ]
    if operation.instructions:
        parts.append(operation.instructions)
    return _with_file_refs("\n\n".join(parts), operation.file_refs)


def _command_summary(tool_input: dict[str, Any]) -> str:
    """Name only the executable; command arguments and environment may hold credentials."""
    command = tool_input.get("command")
    for token in command.split() if isinstance(command, str) else ():
        if "=" in token:
            continue
        return token.rsplit("/", 1)[-1] if _EXECUTABLE.fullmatch(token) else "command"
    return "command"


class ClaudeAgentBackend:
    """Run Nexus operations through the Claude Python Agent SDK."""

    def __init__(
        self, client_factory: Callable[[ClaudeAgentOptions], Any] = ClaudeSDKClient
    ) -> None:
        self._client_factory = client_factory
        self._observed: dict[str, JobError | OperationResult] = {}
        self._changed: dict[str, list[str]] = {}
        self.descriptor = BackendDescriptor(
            backend_id="claude",
            display_name="Claude Agent",
            description="Claude via the Claude Agent SDK",
            capabilities=BackendCapabilities(
                operations=frozenset({"turn", "fork", "review"}),
                cancellation=True,
                graceful_interrupt=True,
                session_continuation=True,
                session_fork=True,
                input_required=True,
                structured_output=True,
                sandbox_modes=frozenset({"read_only", "workspace_write", "danger_full_access"}),
                review_targets=frozenset({"working_tree", "branch", "commit"}),
                review_deliveries=frozenset({"inline", "detached"}),
            ),
        )

    async def check_availability(self, workspace: Workspace) -> BackendAvailability:
        """Report SDK and CLI presence without making an authentication claim."""
        del workspace
        override = get_agent_env("claude", "PATH")
        if override is not None and shutil.which(override) is None:
            return BackendAvailability(
                available=False,
                reason="NEXUS_CLAUDE_PATH does not point to an executable",
                setup_guidance="Unset NEXUS_CLAUDE_PATH to use the CLI bundled with the SDK",
            )
        if override is None:
            try:
                SubprocessCLITransport("", ClaudeAgentOptions())._find_cli()
            except CLINotFoundError:
                return BackendAvailability(
                    available=False,
                    reason="Claude CLI is not available",
                    setup_guidance="Install Claude CLI or set NEXUS_CLAUDE_PATH",
                )
        return BackendAvailability(
            available=True,
            authenticated=None,
            version=f"sdk {claude_agent_sdk.__version__} / cli {__cli_version__}",
        )

    async def resolve_execution_config(
        self, requested: RequestedExecutionConfig, workspace: Workspace
    ) -> ResolvedExecutionConfig:
        """Default to read-only, ask-on-request; reuse existing Nexus limits as fallback."""
        del workspace
        defaults = get_runner_defaults("claude")
        assert defaults.max_retries is not None
        assert defaults.retry_base_delay is not None
        assert defaults.retry_max_delay is not None
        return ResolvedExecutionConfig.from_requested(
            requested,
            backend_defaults=ExecutionConfigValues(
                sandbox="read_only", approval_policy="on_request"
            ),
            fallback_defaults=ExecutionConfigValues(
                model=defaults.model,
                timeout_seconds=defaults.timeout,
                output_limit_bytes=defaults.output_limit,
                retry_policy=RetryPolicy(
                    max_attempts=defaults.max_retries,
                    base_delay_seconds=defaults.retry_base_delay,
                    max_delay_seconds=defaults.retry_max_delay,
                ),
            ),
            fallback_source="fallback",
        )

    async def execute(
        self, operation: AgentOperation, context: BackendExecutionContext
    ) -> OperationResult:
        """Execute exactly one admitted operation in one SDK process."""
        config = context.resolved_config
        review = isinstance(operation, ReviewOperation)
        sandbox: SandboxMode = "read_only" if review else (config.sandbox or "read_only")
        self._require_sandbox_platform(sandbox)
        session_id = next(
            (ref.value for ref in context.job.source_checkpoint if ref.kind == _SESSION), None
        )
        if isinstance(operation, ForkOperation) or (
            isinstance(operation, ReviewOperation) and operation.delivery == "detached"
        ):
            session_id = await self._fork(session_id, context)

        match operation:
            case ForkOperation(prompt=None):
                return self._fork_result(context, session_id)
            case ForkOperation(prompt=str(prompt), file_refs=refs):
                await self._run_turn(
                    context,
                    prompt=_with_file_refs(prompt, refs),
                    sandbox=sandbox,
                    resume=session_id,
                )
                return self._fork_result(context, session_id)
            case ReviewOperation():
                if session_id is None:
                    raise _failure("session_not_found", "No Claude session exists to review")
                final = await self._run_turn(
                    context,
                    prompt=_review_prompt(operation),
                    sandbox=sandbox,
                    resume=session_id,
                    output_schema=ReviewResult.model_json_schema(),
                    review=True,
                )
                try:
                    return ReviewResult.model_validate(
                        (
                            final.structured_output
                            if isinstance(final.structured_output, dict)
                            else {}
                        )
                        | {"target": operation.target, "delivery": operation.delivery}
                    )
                except ValidationError:
                    raise self._observe(
                        context,
                        _failure("structured_output_invalid", "Claude returned an invalid review"),
                    ) from None
            case TurnOperation():
                return await self._turn(operation, context, sandbox, session_id)
        raise _failure("unsupported_capability", "Claude Agent does not support this operation")

    async def _turn(
        self,
        operation: TurnOperation,
        context: BackendExecutionContext,
        sandbox: SandboxMode,
        session_id: str | None,
    ) -> TurnResult:
        """Run a turn and translate its SDK result to the Nexus result contract."""
        config = context.resolved_config
        # The stored schema is frozen; the model serializer thaws it to plain JSON.
        schema = operation.model_dump().get("output_schema")
        final = await self._run_turn(
            context,
            prompt=_with_file_refs(operation.prompt, operation.file_refs),
            sandbox=sandbox,
            resume=session_id,
            output_schema=schema,
        )
        if schema is not None:
            if final.structured_output is None:
                raise self._observe(
                    context,
                    _failure("structured_output_invalid", "Claude returned no structured output"),
                )
            try:
                jsonschema.validate(final.structured_output, schema)
            except (jsonschema.ValidationError, jsonschema.SchemaError):
                raise self._observe(
                    context,
                    _failure(
                        "structured_output_invalid",
                        "Claude returned structured output that violates the requested schema",
                    ),
                ) from None
        return TurnResult(
            message=_truncate(final.result or "", config.output_limit_bytes),
            structured_output=final.structured_output,
            changed_files=tuple(self._changed.pop(context.job.job_id, ())),
            usage=final.usage or {},
        )

    @staticmethod
    async def _fork(session_id: str | None, context: BackendExecutionContext) -> str:
        """Create and record a local SDK fork without starting a model conversation."""
        if session_id is None:
            raise _failure("session_not_found", "No Claude session exists to fork")
        forked = await asyncio.to_thread(
            fork_session, session_id, directory=str(context.workspace.canonical_path)
        )
        await context.record_provider_reference(
            ProviderReference(kind=_SESSION, value=forked.session_id)
        )
        return forked.session_id

    @staticmethod
    def _fork_result(context: BackendExecutionContext, session_id: str | None) -> ForkResult:
        """Build the child-session result once its provider reference is available."""
        session = context.session
        if session is None or session.parent_session_id is None or session_id is None:
            raise _failure("internal_error", "Fork job has no parent Nexus session")
        return ForkResult(
            session=session,
            provider_reference=ProviderReference(kind=_SESSION, value=session_id),
            parent_session_id=session.parent_session_id,
        )

    async def reconcile(
        self,
        provider_state: tuple[ProviderReference, ...],
        context: BackendExecutionContext,
    ) -> ReconciliationOutcome:
        """Return unknown until Task 8 implements observed outcome recovery."""
        del provider_state, context
        return UnknownReconciliationOutcome(
            error=JobError(
                code="outcome_unknown",
                message="Claude process state was lost",
                retry_disposition="reconcile_required",
                recoverable=True,
            )
        )

    async def close(self) -> None:
        """Drop remembered outcomes after all one-turn clients have exited."""
        self._observed.clear()
        self._changed.clear()

    @staticmethod
    def _require_sandbox_platform(sandbox: SandboxMode) -> None:
        if sandbox == "workspace_write" and sys.platform not in ("darwin", "linux"):
            raise _failure(
                "backend_unavailable",
                "Claude Agent workspace_write requires the macOS or Linux sandbox",
            )

    def _observe(self, context: BackendExecutionContext, failure: BackendFailure) -> BackendFailure:
        """Remember definitive failure for worker reconciliation after a provider reference."""
        self._observed[context.job.job_id] = failure.error
        return failure

    async def _run_turn(
        self,
        context: BackendExecutionContext,
        *,
        prompt: str,
        sandbox: SandboxMode,
        resume: str | None,
        output_schema: Any = None,
        review: bool = False,
    ) -> ResultMessage:
        config = context.resolved_config
        approval: ApprovalPolicy = config.approval_policy or "on_request"
        options = build_options(
            workspace=context.workspace.canonical_path,
            sandbox=sandbox,
            profile=get_claude_settings_profile(),
            can_use_tool=self._permission_handler(context, sandbox, approval, review),
            pre_tool_use=self._pre_tool_use(context, sandbox, approval, review),
            resume=resume,
            model=config.model,
            output_schema=output_schema,
            cli_path=get_agent_env("claude", "PATH"),
        )
        recorded: set[str] = set()
        pending: dict[str, str] = {}
        finals: list[ResultMessage] = []

        async def drain(client: Any) -> None:
            async for message in client.receive_response():
                found = await self._translate(message, context, recorded, pending)
                if found is not None:
                    finals.append(found)

        async with self._client_factory(options) as client:
            await client.query(prompt)
            draining = asyncio.create_task(drain(client))
            try:
                await self._await_drain_or_cancel(client, draining, context)
            finally:
                if not draining.done():
                    draining.cancel()
                await asyncio.gather(draining, return_exceptions=True)
        if not finals:
            raise _failure("outcome_unknown", "Claude ended without a result", "reconcile_required")
        return finals[-1]

    @staticmethod
    async def _await_drain_or_cancel(
        client: Any, draining: asyncio.Task[None], context: BackendExecutionContext
    ) -> None:
        """Race the stream against control; lease loss and shutdown cancel this task."""
        while True:
            control = asyncio.create_task(context.wait_for_control())
            try:
                done, _ = await asyncio.wait(
                    {draining, control}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                if not control.done():
                    control.cancel()
                await asyncio.gather(control, return_exceptions=True)
            if control in done and isinstance(control.result(), CancelRequested):
                await client.interrupt()
                await draining
                raise asyncio.CancelledError
            if draining.done():
                draining.result()
                return

    async def _translate(
        self,
        message: Any,
        context: BackendExecutionContext,
        recorded: set[str],
        pending: dict[str, str],
    ) -> ResultMessage | None:
        session_id = getattr(message, "session_id", None)
        if isinstance(session_id, str) and session_id and not recorded:
            recorded.add(session_id)
            await context.record_provider_reference(
                ProviderReference(kind=_SESSION, value=session_id)
            )
        match message:
            case AssistantMessage(content=blocks) | UserMessage(content=blocks):
                for block in blocks if isinstance(blocks, list) else ():
                    await self._translate_block(block, context, pending)
            case ResultMessage():
                return message
        return None

    async def _translate_block(
        self, block: Any, context: BackendExecutionContext, pending: dict[str, str]
    ) -> None:
        match block:
            case TextBlock(text=value):
                await context.emit_output_delta(value)
            case ToolUseBlock(name="Bash", input=tool_input):
                await context.emit(
                    BackendEvent(type="command", payload={"command": _command_summary(tool_input)})
                )
            case ToolUseBlock(id=tool_use_id, name=name, input=tool_input) if name in WRITE_TOOLS:
                path = tool_input.get("file_path") or tool_input.get("notebook_path")
                if isinstance(path, str) and path:
                    pending[tool_use_id] = path[:4096]
            case ToolResultBlock(tool_use_id=tool_use_id, is_error=is_error):
                path = pending.pop(tool_use_id, None)
                if path is not None and not is_error:
                    self._changed.setdefault(context.job.job_id, []).append(path)
                    await context.emit(BackendEvent(type="file_change", payload={"path": path}))

    @staticmethod
    def _pre_tool_use(
        context: BackendExecutionContext,
        sandbox: SandboxMode,
        approval: ApprovalPolicy,
        review: bool,
    ) -> Any:
        """Gate all tools, including those settings allow before can_use_tool runs."""

        async def gate(
            input_data: dict[str, Any], tool_use_id: str | None, hook_context: Any
        ) -> dict[str, Any]:
            del tool_use_id, hook_context
            tool_input = input_data.get("tool_input")
            verdict = decide(
                str(input_data.get("tool_name") or ""),
                tool_input if isinstance(tool_input, dict) else {},
                sandbox=sandbox,
                approval=approval,
                workspace=context.workspace.canonical_path,
                review=review,
            )
            if verdict == "allow":
                return {}
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": verdict,
                    "permissionDecisionReason": "Nexus sandbox policy",
                }
            }

        return gate

    def _permission_handler(
        self,
        context: BackendExecutionContext,
        sandbox: SandboxMode,
        approval: ApprovalPolicy,
        review: bool,
    ) -> Any:
        """Resolve SDK tool approval through persisted Nexus permission requests."""

        async def handler(
            tool: str, tool_input: dict[str, Any], permission: ToolPermissionContext
        ) -> PermissionResultAllow | PermissionResultDeny:
            del permission
            verdict = decide(
                tool,
                tool_input,
                sandbox=sandbox,
                approval=approval,
                workspace=context.workspace.canonical_path,
                review=review,
            )
            if verdict == "allow":
                return PermissionResultAllow()
            if verdict == "deny":
                return PermissionResultDeny(message="Denied by Nexus sandbox policy")
            path = tool_input.get("notebook_path" if tool == "NotebookEdit" else "file_path")
            scope = f"{tool}:{path}" if isinstance(path, str) and path else tool
            if len(scope) > 2048:
                return PermissionResultDeny(message="Permission scope exceeds Nexus limit")
            response = await context.request_input(
                PermissionRequest(
                    prompt=f"Claude wants to use {tool}",
                    risk="Nexus sandbox policy requires approval",
                    requested=frozenset({scope}),
                )
            )
            if getattr(response, "granted", None):
                return PermissionResultAllow()
            return PermissionResultDeny(message="Denied by user")

        return handler
