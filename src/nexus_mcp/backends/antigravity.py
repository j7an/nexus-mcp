"""Antigravity backend: one Antigravity SDK harness process per prompt call."""

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import anyio
from fastmcp.exceptions import ToolError
from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.types import (
    AntigravityConnectionError,
    AntigravityExecutionError,
    AntigravityValidationError,
    BuiltinTools,
    RunCommandConfig,
    SessionContinuationMode,
    StopReason,
)
from pydantic import ValidationError

from nexus_mcp.types import BackendInfo, PromptRequest, PromptResult

__all__ = ["Agent", "info", "run"]

NAME = "antigravity"
_STORE = Path.home() / ".nexus-mcp" / "antigravity"
_READ_TOOLS = (
    BuiltinTools.VIEW_FILE,
    BuiltinTools.LIST_DIR,
    BuiltinTools.SEARCH_DIR,
    BuiltinTools.FIND_FILE,
    BuiltinTools.FINISH,
)
_WRITE_TOOLS = _READ_TOOLS + (
    BuiltinTools.CREATE_FILE,
    BuiltinTools.EDIT_FILE,
    BuiltinTools.RUN_COMMAND,
)


def _config(req: PromptRequest, conversation_id: str) -> LocalAgentConfig:
    if req.profile == "full_access":
        tools = tuple(t for t in BuiltinTools if t != BuiltinTools.ASK_QUESTION)
        policies = [policy.allow_all()]
    else:
        tools = _WRITE_TOOLS if req.profile == "workspace_write" else _READ_TOOLS
        policies = [
            policy.deny_all(),
            *(policy.allow(t.value) for t in tools),
            *policy.workspace_only([req.cwd]),
        ]
    capabilities = CapabilitiesConfig(
        enabled_tools=list(tools),
        run_command_config=(
            RunCommandConfig(enable_sandbox=True) if req.profile == "workspace_write" else None
        ),
    )
    mode = SessionContinuationMode.RESUME if req.session_id else SessionContinuationMode.CREATE_ONLY
    try:
        return LocalAgentConfig(
            workspaces=[str(req.cwd)],
            save_dir=str(_STORE),
            conversation_id=conversation_id,
            session_continuation_mode=mode,
            model=req.model,
            capabilities=capabilities,
            policies=policies,
        )
    except ValidationError:
        raise ToolError("session_id is not a valid Antigravity conversation id") from None


def _ignore_session(_session_id: str) -> None:
    return None


@contextlib.asynccontextmanager
async def _session(config: LocalAgentConfig) -> AsyncIterator[Agent]:
    """Finish pending startup before shielded teardown on every exit."""
    agent = Agent(config)
    startup = asyncio.ensure_future(agent.__aenter__())
    try:
        try:
            await asyncio.shield(startup)
        except AntigravityValidationError:
            raise ToolError(
                "Antigravity needs credentials: set GEMINI_API_KEY, or configure Vertex "
                "(GOOGLE_GENAI_USE_VERTEXAI with project/location or an API key)"
            ) from None
        except RuntimeError:
            raise ToolError(
                "Antigravity could not start or resume the session "
                "(unknown session_id, or the harness failed to start)"
            ) from None
        yield agent
    finally:
        with anyio.CancelScope(shield=True):
            if not startup.done():
                with contextlib.suppress(Exception):
                    await startup
            # SDK disconnect() blocks on process.wait(timeout=180). Move teardown to a
            # worker thread if a shutdown hang is observed.
            await agent.__aexit__(None, None, None)


async def info() -> BackendInfo:
    """Report installation without starting a harness or listing models."""
    return BackendInfo(name=NAME, installed=True, models=None)


async def run(
    req: PromptRequest, on_session: Callable[[str], None] = _ignore_session
) -> PromptResult:
    """Run one new or resumed Antigravity turn with the requested permission profile."""
    if req.fork:
        raise ToolError("Antigravity does not support fork; omit fork to continue the session")
    conversation_id = req.session_id or str(uuid.uuid4())
    config = _config(req, conversation_id)
    try:
        _STORE.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ToolError(
            "Antigravity session store ~/.nexus-mcp/antigravity is not writable"
        ) from None
    async with _session(config) as agent:
        on_session(conversation_id)
        if req.profile == "workspace_write" and not (
            agent.sandbox_status is not None and agent.sandbox_status.available
        ):
            raise ToolError(
                "Antigravity OS sandbox unavailable; workspace_write cannot confine commands. "
                "Use read_only, or full_access to run unsandboxed"
            )
        try:
            try:
                response = await agent.chat(req.prompt)
            except ValueError:
                raise ToolError(
                    "Antigravity rejected the prompt (empty after trimming whitespace)"
                ) from None
            text = await response.text()
            if response.stop_reason == StopReason.QUOTA_EXHAUSTED:
                raise ToolError("Antigravity quota exhausted; retry later")
            if agent.conversation_id != conversation_id:
                raise ToolError(
                    "Antigravity returned an unexpected conversation id; "
                    "refusing to report an unresumable session"
                )
        except (AntigravityConnectionError, AntigravityExecutionError) as error:
            raise ToolError(
                f"Antigravity run failed ({type(error).__name__}); check credentials and model"
            ) from None
        usage = response.usage_metadata
        return PromptResult(
            backend=NAME,
            session_id=conversation_id,
            output=text,
            usage=None if usage is None else usage.model_dump(mode="json", exclude_none=True),
        )
