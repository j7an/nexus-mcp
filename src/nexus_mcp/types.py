import math
from typing import Annotated, Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

type ExecutionMode = Literal["default", "yolo"]
type LogLevel = Literal["debug", "info", "warning", "error"]


class LogEmitter(Protocol):
    """Callback protocol for emitting log messages from runners.

    Any async callable matching (level: LogLevel, message: str) -> None satisfies
    this protocol. Runners accept an optional LogEmitter; server.py provides one
    that sends to both MCP clients and Python's logging module.
    """

    async def __call__(self, level: LogLevel, message: str) -> None: ...


# Shared Annotated type aliases — single source of truth for field constraints.
# Used across OperationalDefaults, SessionPreferences, PromptRequest, AgentTask.
Timeout = Annotated[int | None, Field(ge=1)]
OutputLimit = Annotated[int | None, Field(ge=1)]
MaxRetries = Annotated[int | None, Field(ge=1)]
ModelName = Annotated[str | None, Field(min_length=1)]
Delay = Annotated[float | None, Field(ge=0)]


class OperationalDefaults(BaseModel, frozen=True):
    """Shape of operational settings at any tier.

    All fields are None-able — None means "not set at this tier".
    After merging all tiers, HARDCODED_DEFAULTS guarantees non-None for required fields.
    """

    timeout: int | None = Field(default=None, ge=1)
    output_limit: int | None = Field(default=None, ge=1)
    max_retries: int | None = Field(default=None, ge=1)
    retry_base_delay: Annotated[float, Field(ge=0)] | None = None
    retry_max_delay: Annotated[float, Field(ge=0)] | None = None
    tool_timeout: Annotated[float, Field(ge=0)] | None = None  # raw value; 0 → None in getter
    cli_detection_timeout: int | None = Field(default=None, ge=1)
    execution_mode: ExecutionMode | None = None
    model: str | None = Field(default=None, min_length=1)

    @field_validator("retry_base_delay", "retry_max_delay", "tool_timeout", mode="after")
    @classmethod
    def reject_non_finite(cls, v: float | None) -> float | None:
        """Safety net for programmatic construction.
        Env vars are validated manually in _read_global_env_defaults() with ConfigurationError."""
        if v is not None and not math.isfinite(v):
            raise ValueError(f"must be a finite number, got {v}")
        return v


class SessionPreferences(BaseModel):
    """Session-scoped preferences set by set_preferences and applied by prompt/batch_prompt."""

    model_config = ConfigDict(frozen=True)

    execution_mode: ExecutionMode | None = None
    model: ModelName = None
    max_retries: MaxRetries = None
    output_limit: OutputLimit = None
    timeout: Timeout = None
    retry_base_delay: Delay = None
    retry_max_delay: Delay = None


DEFAULT_MAX_CONCURRENCY = 3
MAX_PROMPT_LENGTH = 131072  # 128KB character limit — conservative guard against ARG_MAX


class PromptRequest(BaseModel):
    cli: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    context: dict[str, Any] = Field(default_factory=dict)
    execution_mode: ExecutionMode = Field(
        default="default",
        description="Execution mode: 'default' (safe) or 'yolo'",
    )
    model: str | None = Field(
        default=None,
        description="Optional model name (uses CLI default if not specified)",
        min_length=1,
    )
    file_refs: list[str] = Field(
        default_factory=list,
        description="Optional file paths for agent context (appended to prompt)",
    )

    @field_validator("file_refs")
    @classmethod
    def no_control_chars_in_paths(cls, v: list[str]) -> list[str]:
        for i, path in enumerate(v):
            if any(c in path for c in "\x00\n\r"):
                raise ValueError(
                    f"file_refs[{i}] contains control characters (null/newline/carriage-return)"
                )
        return v

    max_retries: MaxRetries = Field(
        default=None,
        description="Max retry attempts for transient errors (None uses NEXUS_RETRY_MAX_ATTEMPTS)",
    )
    output_limit: OutputLimit = Field(
        default=None,
        description="Max output bytes (None uses NEXUS_OUTPUT_LIMIT_BYTES)",
    )
    timeout: Timeout = Field(
        default=None,
        description="Subprocess timeout seconds (None uses NEXUS_TIMEOUT_SECONDS)",
    )
    retry_base_delay: Delay = Field(
        default=None,
        description="Base delay seconds for exponential backoff (None uses NEXUS_RETRY_BASE_DELAY)",
    )
    retry_max_delay: Delay = Field(
        default=None,
        description="Max delay cap for backoff in seconds (None uses NEXUS_RETRY_MAX_DELAY)",
    )


class AgentResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    cli: str
    output: str
    raw_output: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_metadata(self, **updates: Any) -> Self:
        """Return a copy with updated metadata keys."""
        metadata = self.metadata.copy()
        metadata.update(updates)
        return self.model_copy(update={"metadata": metadata})


class SubprocessResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    stdout: str
    stderr: str
    returncode: int


class AgentTask(BaseModel):
    """Per-task input for batch_prompt."""

    cli: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    label: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    execution_mode: ExecutionMode | None = None  # None = use session preference or "default"
    model: ModelName = None
    max_retries: MaxRetries = None
    output_limit: OutputLimit = None
    timeout: Timeout = None
    retry_base_delay: Delay = None
    retry_max_delay: Delay = None

    def to_request(self) -> "PromptRequest":
        """Convert this task to a PromptRequest for runner execution."""
        return PromptRequest(
            cli=self.cli,
            prompt=self.prompt,
            context=self.context,
            execution_mode=self.execution_mode or "default",  # safety net: None → "default"
            model=self.model,
            max_retries=self.max_retries,
            output_limit=self.output_limit,
            timeout=self.timeout,
            retry_base_delay=self.retry_base_delay,
            retry_max_delay=self.retry_max_delay,
        )


class AgentTaskResult(BaseModel):
    """Per-task output for batch_prompt — exactly one of output/error is set."""

    model_config = ConfigDict(frozen=True)

    label: str
    output: str | None = None
    error: str | None = None
    error_type: str | None = None  # e.g. "ParseError", "SubprocessError"

    @model_validator(mode="after")
    def exactly_one_of_output_or_error(self) -> Self:
        if self.output is not None and self.error is not None:
            raise ValueError("output and error are mutually exclusive")
        if self.output is None and self.error is None:
            raise ValueError("one of output or error must be set")
        return self

    @property
    def success(self) -> bool:
        return self.output is not None

    @property
    def formatted_error(self) -> str:
        """Error message with [ErrorType] prefix, for use in ToolError messages."""
        prefix = f"[{self.error_type}] " if self.error_type else ""
        return f"{prefix}{self.error}"


class MultiPromptResponse(BaseModel):
    """Aggregate response from batch_prompt with auto-computed counts."""

    model_config = ConfigDict(frozen=True)

    results: list[AgentTaskResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return len(self.results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def succeeded(self) -> int:
        return sum(1 for r in self.results if r.success)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def failed(self) -> int:
        return self.total - self.succeeded
