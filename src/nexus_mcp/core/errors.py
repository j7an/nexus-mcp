"""Stable exceptions for framework-independent job-domain failures."""


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
