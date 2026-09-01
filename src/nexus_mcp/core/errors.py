"""Stable exceptions for framework-independent job-domain failures."""

__all__ = [
    "AccessDeniedError",
    "BackendDisabledError",
    "BackendUnknownError",
    "IdempotencyConflictError",
    "InputAlreadyResolvedError",
    "InputNotFoundError",
    "InvalidJobTransitionError",
    "JobNotFoundError",
    "NexusCoreError",
    "SessionBusyError",
    "SessionNotFoundError",
    "StaleLeaseError",
    "UnsupportedCapabilityError",
    "WorkspaceInvalidError",
]


class NexusCoreError(Exception):
    """Base class for failures communicated by the Nexus job core."""

    code = "nexus_core_error"


class InvalidJobTransitionError(NexusCoreError):
    """Raised when a job attempts a state transition outside the public state machine."""

    code = "invalid_job_transition"

    def __init__(self, *, current: str, target: str) -> None:
        self.current = current
        self.target = target
        super().__init__(f"Cannot transition job from {current!r} to {target!r}")


class BackendUnknownError(NexusCoreError):
    """Raised when a backend identifier is not registered."""

    code = "backend_unknown"

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        super().__init__(f"Unknown backend: {backend_id}")


class BackendDisabledError(NexusCoreError):
    """Raised when a known backend is administratively disabled."""

    code = "backend_disabled"

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        super().__init__(f"Backend is disabled: {backend_id}")


class UnsupportedCapabilityError(NexusCoreError):
    """Raised when admission requests an unadvertised backend capability."""

    code = "unsupported_capability"

    def __init__(self, backend_id: str, capability: str) -> None:
        self.backend_id = backend_id
        self.capability = capability
        super().__init__(f"Backend {backend_id} does not support {capability}")


class WorkspaceInvalidError(NexusCoreError):
    """Raised when a workspace cannot be resolved or admitted safely."""

    code = "workspace_invalid"

    def __init__(self, workspace: str, reason: str = "workspace is invalid") -> None:
        self.workspace = workspace
        self.reason = reason
        super().__init__(f"Invalid workspace {workspace}: {reason}")


class SessionNotFoundError(NexusCoreError):
    """Raised when no authorized session matches an identifier."""

    code = "session_not_found"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionBusyError(NexusCoreError):
    """Raised when a session already has a nonterminal job."""

    code = "session_busy"

    def __init__(self, session_id: str, active_job_id: str) -> None:
        self.session_id = session_id
        self.active_job_id = active_job_id
        super().__init__(f"Session {session_id} already has active job {active_job_id}")


class JobNotFoundError(NexusCoreError):
    """Raised when no authorized job matches an identifier."""

    code = "job_not_found"

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Job not found: {job_id}")


class InputNotFoundError(NexusCoreError):
    """Raised when no pending input matches a job-scoped identifier."""

    code = "input_not_found"

    def __init__(self, job_id: str, input_id: str) -> None:
        self.job_id = job_id
        self.input_id = input_id
        super().__init__(f"Input {input_id} not found for job {job_id}")


class AccessDeniedError(NexusCoreError):
    """Raised when a trusted caller cannot create the requested object."""

    code = "access_denied"

    def __init__(self, reason: str = "Access denied") -> None:
        self.reason = reason
        super().__init__(reason)


class IdempotencyConflictError(NexusCoreError):
    """Raised when one idempotency key names a different admitted request."""

    code = "idempotency_conflict"

    def __init__(self, idempotency_key: str) -> None:
        self.idempotency_key = idempotency_key
        super().__init__(f"Idempotency key conflicts with an existing request: {idempotency_key}")


class InputAlreadyResolvedError(NexusCoreError):
    """Raised when a second input response conflicts with the committed response."""

    code = "input_already_resolved"

    def __init__(self, job_id: str, input_id: str) -> None:
        self.job_id = job_id
        self.input_id = input_id
        super().__init__(f"Input {input_id} for job {job_id} is already resolved")


class StaleLeaseError(NexusCoreError):
    """Raised when a worker mutation uses an obsolete lease generation."""

    code = "stale_lease"

    def __init__(self, job_id: str, generation: int) -> None:
        self.job_id = job_id
        self.generation = generation
        super().__init__(f"Lease generation {generation} is stale for job {job_id}")
