"""Temporary framework-independent backend over the existing runner implementations."""

from pathlib import Path

from nexus_mcp.backends.base import (
    BackendExecutionContext,
    BackendFailure,
    ReconciliationOutcome,
    UnknownReconciliationOutcome,
)
from nexus_mcp.cli_detector import detect_cli, get_cli_version
from nexus_mcp.config import get_runner_defaults
from nexus_mcp.core import (
    AgentOperation,
    BackendAvailability,
    BackendCapabilities,
    BackendDescriptor,
    BackendEvent,
    ExecutionConfigValues,
    JobError,
    JobErrorCode,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    RetryPolicy,
    SandboxMode,
    TurnOperation,
    TurnResult,
    Workspace,
)
from nexus_mcp.exceptions import (
    CLINotFoundError,
    ParseError,
    RetryableError,
    SubprocessError,
    SubprocessTimeoutError,
)
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.types import ExecutionMode, LogLevel, PromptRequest

__all__ = ["LegacyRunnerBackend", "legacy_backends"]


class LegacyRunnerBackend:
    """Expose one existing runner through the durable backend protocol."""

    def __init__(self, backend_id: str) -> None:
        runner_class = RunnerFactory.get_runner_class(backend_id)
        sandbox_modes: frozenset[SandboxMode] = (
            frozenset({"danger_full_access"})
            if "yolo" in runner_class._SUPPORTED_MODES
            else frozenset()
        )
        self._backend_id = backend_id
        self.descriptor = BackendDescriptor(
            backend_id=backend_id,
            display_name=backend_id.replace("_", " ").title(),
            description="Temporary bridge to the legacy Nexus runner",
            capabilities=BackendCapabilities(
                operations=frozenset({"turn"}),
                cancellation=False,
                graceful_interrupt=False,
                session_fork=False,
                sandbox_modes=sandbox_modes,
            ),
        )

    async def check_availability(self, workspace: Workspace) -> BackendAvailability:
        """Observe current legacy CLI reachability without changing registration."""
        del workspace
        detected = detect_cli(self._backend_id)
        if not detected.found:
            return BackendAvailability(
                available=False,
                reason=f"{self._backend_id} is not available",
                setup_guidance=f"Install and configure {self._backend_id}",
            )
        return BackendAvailability(
            available=True,
            version=get_cli_version(self._backend_id),
        )

    async def resolve_execution_config(
        self,
        requested: RequestedExecutionConfig,
        workspace: Workspace,
    ) -> ResolvedExecutionConfig:
        """Resolve legacy Nexus settings only as the final fallback layer."""
        del workspace
        defaults = get_runner_defaults(self._backend_id)
        assert defaults.max_retries is not None
        assert defaults.retry_base_delay is not None
        assert defaults.retry_max_delay is not None
        fallback = ExecutionConfigValues(
            model=defaults.model,
            timeout_seconds=defaults.timeout,
            output_limit_bytes=defaults.output_limit,
            retry_policy=RetryPolicy(
                max_attempts=defaults.max_retries,
                base_delay_seconds=defaults.retry_base_delay,
                max_delay_seconds=defaults.retry_max_delay,
            ),
        )
        return ResolvedExecutionConfig.from_requested(
            requested,
            backend_defaults=ExecutionConfigValues(),
            fallback_defaults=fallback,
            fallback_source="legacy_nexus_fallback",
        )

    async def execute(
        self,
        operation: AgentOperation,
        context: BackendExecutionContext,
    ) -> TurnResult:
        """Execute one admitted turn with exactly one legacy runner attempt."""
        if not isinstance(operation, TurnOperation):
            raise self._policy_failure(f"operation {operation.kind}")
        execution_mode = self._execution_mode(context.resolved_config)
        if self._backend_id == "opencode_server":
            admitted = context.workspace.canonical_path.resolve()
            if admitted != Path.cwd().resolve():
                raise BackendFailure(
                    JobError(
                        code="workspace_unsupported",
                        message=(
                            "The opencode_server legacy backend cannot execute outside "
                            "the Nexus process workspace"
                        ),
                    ),
                    "terminal",
                )

        request = self._prompt_request(operation, context, execution_mode)

        async def emit_log(level: LogLevel, message: str) -> None:
            await context.emit(
                BackendEvent(type="log", payload={"level": level, "message": message})
            )

        async def emit_progress(progress: float, total: float, message: str) -> None:
            await context.emit(
                BackendEvent(
                    type="progress",
                    payload={"progress": progress, "total": total, "message": message},
                )
            )

        try:
            runner = RunnerFactory.create(self._backend_id)
            response = await runner.run(request, emitter=emit_log, progress=emit_progress)
        except Exception as exc:
            raise self._normalized_failure(exc) from None
        return TurnResult(message=response.output)

    async def reconcile(
        self,
        provider_state: tuple[ProviderReference, ...],
        context: BackendExecutionContext,
    ) -> ReconciliationOutcome:
        """Return unknown because legacy runners expose no safe replay or lookup contract."""
        del provider_state, context
        return UnknownReconciliationOutcome(
            error=JobError(
                code="outcome_unknown",
                message="Legacy runner state cannot be reconciled safely",
                retry_disposition="reconcile_required",
                recoverable=True,
            )
        )

    async def close(self) -> None:
        """Legacy runner instances have no backend-owned lifecycle resources."""

    def _execution_mode(self, config: ResolvedExecutionConfig) -> ExecutionMode:
        sandbox = config.sandbox
        approval = config.approval_policy
        if sandbox == "danger_full_access" and approval == "never":
            if "danger_full_access" in self.descriptor.capabilities.sandbox_modes:
                return "yolo"
            raise self._policy_failure("danger_full_access sandbox")
        if sandbox is not None or approval not in {None, "provider_default", "on_request"}:
            raise self._policy_failure("sandbox or approval policy")
        return "default"

    def _prompt_request(
        self,
        operation: TurnOperation,
        context: BackendExecutionContext,
        execution_mode: ExecutionMode,
    ) -> PromptRequest:
        config = context.resolved_config
        retry_policy = config.retry_policy
        return PromptRequest(
            cli=self._backend_id,
            prompt=operation.prompt,
            context=dict(operation.context),
            file_refs=list(operation.file_refs),
            execution_mode=execution_mode,
            model=config.model,
            max_retries=1,
            output_limit=config.output_limit_bytes,
            timeout=config.timeout_seconds,
            retry_base_delay=(None if retry_policy is None else retry_policy.base_delay_seconds),
            retry_max_delay=(None if retry_policy is None else retry_policy.max_delay_seconds),
            cwd=context.workspace.canonical_path,
        )

    def _policy_failure(self, capability: str) -> BackendFailure:
        return BackendFailure(
            JobError(
                code="unsupported_capability",
                message=f"Legacy backend {self._backend_id} does not support {capability}",
            ),
            "terminal",
        )

    def _normalized_failure(self, exc: Exception) -> BackendFailure:
        exception_type = type(exc).__name__
        if len(exception_type) > 128 or not exception_type.isidentifier():
            exception_type = "Exception"
        details = {"legacy_exception_type": exception_type}
        if isinstance(exc, RetryableError):
            error = JobError(
                code="provider_failed",
                message=f"Legacy backend {self._backend_id} failed with a retryable error",
                retry_disposition="safe_to_retry",
                recoverable=True,
                details=details,
            )
            return BackendFailure(error, "safe_to_retry")
        if isinstance(exc, SubprocessTimeoutError):
            code: JobErrorCode = "timeout"
            message = f"Legacy backend {self._backend_id} timed out"
        elif isinstance(exc, ParseError):
            code = "structured_output_invalid"
            message = f"Legacy backend {self._backend_id} returned an invalid response"
        elif isinstance(exc, CLINotFoundError):
            code = "backend_unavailable"
            message = f"Legacy backend {self._backend_id} is unavailable"
        elif isinstance(exc, SubprocessError):
            code = "provider_failed"
            message = f"Legacy backend {self._backend_id} process failed"
        else:
            code = "internal_error"
            message = f"Legacy backend {self._backend_id} failed internally"
        error = JobError(
            code=code,
            message=message,
            retry_disposition="terminal",
            recoverable=False,
            details=details,
        )
        return BackendFailure(error, "terminal")


def legacy_backends() -> tuple[LegacyRunnerBackend, ...]:
    """Return one deterministic adapter for every registered legacy runner."""
    return tuple(LegacyRunnerBackend(backend_id) for backend_id in RunnerFactory.list_clis())
