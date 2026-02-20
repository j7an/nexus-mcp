from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

type ExecutionMode = Literal["default", "sandbox", "yolo"]

DEFAULT_MAX_CONCURRENCY = 3


class PromptRequest(BaseModel):
    agent: str
    prompt: str = Field(..., min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    execution_mode: ExecutionMode = Field(
        default="default",
        description="Execution mode: 'default' (safe), 'sandbox', or 'yolo'",
    )
    model: str | None = Field(
        default=None,
        description="Optional model name (uses CLI default if not specified)",
        min_length=1,
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for resumable sessions (Claude Code --resume)",
        min_length=1,
    )
    file_refs: list[str] = Field(
        default_factory=list,
        description="Optional file paths for agent context (appended to prompt)",
    )


class AgentResponse(BaseModel):
    agent: str
    output: str
    raw_output: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def with_metadata(self, **updates: Any) -> "AgentResponse":
        """Return a copy with updated metadata keys."""
        metadata = self.metadata.copy()
        metadata.update(updates)
        return self.model_copy(update={"metadata": metadata})


class SubprocessResult(BaseModel):
    stdout: str
    stderr: str
    returncode: int


class AgentTask(BaseModel):
    """Per-task input for batch_prompt."""

    agent: str
    prompt: str
    label: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    execution_mode: ExecutionMode = "default"
    model: str | None = None

    @field_validator("agent", "prompt")
    @classmethod
    def must_be_non_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v


class AgentTaskResult(BaseModel):
    """Per-task output for batch_prompt — exactly one of output/error is set."""

    label: str
    output: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def exactly_one_of_output_or_error(self) -> "AgentTaskResult":
        if self.output is not None and self.error is not None:
            raise ValueError("output and error are mutually exclusive")
        if self.output is None and self.error is None:
            raise ValueError("one of output or error must be set")
        return self

    @property
    def success(self) -> bool:
        return self.output is not None


class MultiPromptResponse(BaseModel):
    """Aggregate response from batch_prompt with auto-computed counts."""

    results: list[AgentTaskResult]
    total: int = 0
    succeeded: int = 0
    failed: int = 0

    @model_validator(mode="after")
    def compute_counts(self) -> "MultiPromptResponse":
        self.total = len(self.results)
        self.succeeded = sum(1 for r in self.results if r.success)
        self.failed = self.total - self.succeeded
        return self
