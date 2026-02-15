from typing import Any, Literal

from pydantic import BaseModel, Field

type ExecutionMode = Literal["default", "sandbox", "yolo"]


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


class SubprocessResult(BaseModel):
    stdout: str
    stderr: str
    returncode: int
