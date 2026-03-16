from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

type ExecutionMode = Literal["default", "yolo"]


class SessionPreferences(BaseModel):
    """Session-scoped preferences set by set_preferences and applied by prompt/batch_prompt."""

    model_config = ConfigDict(frozen=True)

    execution_mode: ExecutionMode | None = None
    model: str | None = Field(default=None, min_length=1)


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

    max_retries: int | None = Field(
        default=None,
        ge=1,
        description="Max retry attempts for transient errors (None uses NEXUS_RETRY_MAX_ATTEMPTS)",
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


class RunnerInfo(BaseModel, frozen=True):
    """Metadata about a registered runner, returned by list_runners tool."""

    name: str
    type: Literal["cli", "server"]
    provider: str | None
    models: tuple[str, ...]
    available: bool
    default_model: str | None
    execution_modes: tuple[ExecutionMode, ...]


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
    model: str | None = None
    max_retries: int | None = Field(default=None, ge=1)

    def to_request(self) -> "PromptRequest":
        """Convert this task to a PromptRequest for runner execution."""
        return PromptRequest(
            cli=self.cli,
            prompt=self.prompt,
            context=self.context,
            execution_mode=self.execution_mode or "default",  # safety net: None → "default"
            model=self.model,
            max_retries=self.max_retries,
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
