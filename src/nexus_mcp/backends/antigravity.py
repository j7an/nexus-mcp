"""Antigravity backend: one Antigravity SDK harness process per prompt call."""

from pathlib import Path

from fastmcp.exceptions import ToolError
from google.antigravity import CapabilitiesConfig, LocalAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.types import BuiltinTools, RunCommandConfig, SessionContinuationMode
from pydantic import ValidationError

from nexus_mcp.types import PromptRequest

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
