"""FastMCP server exposing the ``prompt`` tool and ``nexus://backends`` resource."""

import asyncio
import json
import re
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from nexus_mcp import backends
from nexus_mcp.types import BackendName, Profile, PromptRequest, PromptResult

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "backends_resource",
    "build_instructions",
    "mcp",
    "run_prompt",
]

DEFAULT_TIMEOUT_SECONDS = 600
# Provider session ids are UUID-like; a leading letter/digit keeps them from reading as CLI flags.
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")


def build_instructions() -> str:
    """Describe the server for the calling model, listing installed backends."""
    names = [name for name in backends.NAMES if backends.installed(name)]
    listed = ", ".join(names) if names else "none (install nexus-mcp[all])"
    return (
        "Delegates tasks to coding agents. "
        f"Installed backends: {listed}. "
        "`cwd` is required (absolute path to the project directory). "
        "Profiles: read_only (default), workspace_write, full_access. "
        "Pass the returned `session_id` to continue a conversation, "
        "or with `fork=true` to branch it."
    )


mcp = FastMCP("nexus-mcp", instructions=build_instructions())


def _resolve_cwd(cwd: str) -> Path:
    if "\x00" in cwd:
        raise ToolError("cwd must not contain NUL bytes")
    path = Path(cwd)
    if not path.is_absolute():
        raise ToolError("cwd must be an absolute path")
    if not path.is_dir():
        raise ToolError("cwd must be an existing directory")
    return path.resolve()


def _check_session(session_id: str | None, fork: bool) -> None:
    if fork and session_id is None:
        raise ToolError("fork=true requires session_id")
    if session_id is not None and not _SESSION_ID.fullmatch(session_id):
        raise ToolError(
            "session_id must be 1-256 letters, digits, '.', '_', ':' or '-', "
            "starting with a letter or digit"
        )


def _timeout_message(
    backend: BackendName, timeout: int, request: PromptRequest, acquired: list[str]
) -> str:
    message = f"{backend} timed out after {timeout}s"
    session = acquired[-1] if acquired else (None if request.fork else request.session_id)
    if session is not None:
        message += f"; continue with session_id={session}"
    return message


@mcp.tool(
    name="prompt",
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=True),
)
async def run_prompt(
    backend: BackendName,
    prompt: Annotated[str, Field(min_length=1)],
    cwd: str,
    profile: Profile = "read_only",
    session_id: str | None = None,
    fork: bool = False,
    model: str | None = None,
    timeout: Annotated[int, Field(ge=1)] = DEFAULT_TIMEOUT_SECONDS,
) -> PromptResult:
    """Run one turn with a coding agent and return its final answer.

    Args:
        backend: Agent to run.
        prompt: Task for the agent.
        cwd: Absolute path of the project directory.
        profile: Permission profile for the agent.
        session_id: Continue this conversation (from a previous result).
        fork: Branch session_id into a new conversation instead of continuing it.
        model: Model id or alias; the provider default when omitted.
        timeout: Seconds before the turn is cancelled.
    """
    _check_session(session_id, fork)
    request = PromptRequest(
        prompt=prompt,
        cwd=_resolve_cwd(cwd),
        profile=profile,
        session_id=session_id,
        fork=fork,
        model=model,
    )
    if not backends.installed(backend):
        raise ToolError(f"Backend '{backend}' is not installed. {backends.install_hint(backend)}")
    acquired: list[str] = []
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            result: PromptResult = await backends.get(backend).run(request, acquired.append)
    except TimeoutError:
        if not deadline.expired():
            raise
        raise ToolError(_timeout_message(backend, timeout, request, acquired)) from None
    return result


@mcp.resource("nexus://backends", mime_type="application/json")
async def backends_resource() -> str:
    """Backend availability, models, and install hints."""
    infos = [await backends.info(name) for name in backends.NAMES]
    return json.dumps([item.model_dump() for item in infos])
