"""Request, result, and backend status models."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

__all__ = ["BackendInfo", "BackendName", "Profile", "PromptRequest", "PromptResult"]

type Profile = Literal["read_only", "workspace_write", "full_access"]
type BackendName = Literal["claude"]


class PromptRequest(BaseModel):
    """One validated agent turn, as handed to a backend."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    cwd: Path
    profile: Profile = "read_only"
    session_id: str | None = None
    fork: bool = False
    model: str | None = None


class PromptResult(BaseModel):
    """Final answer of one agent turn."""

    model_config = ConfigDict(frozen=True)

    backend: str
    session_id: str
    output: str
    usage: dict[str, Any] | None = None


class BackendInfo(BaseModel):
    """Availability of one backend, as listed by nexus://backends."""

    model_config = ConfigDict(frozen=True)

    name: str
    installed: bool
    models: list[str] | None = None
    hint: str | None = None
