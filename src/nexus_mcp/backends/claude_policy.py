"""Deny-by-default permission policy and SDK options for the Claude backend."""

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from claude_agent_sdk import (
    CanUseTool,
    ClaudeAgentOptions,
    HookCallback,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from fastmcp.exceptions import ToolError

from nexus_mcp.types import Profile

if TYPE_CHECKING:
    from claude_agent_sdk.types import SandboxSettings

__all__ = [
    "READ_TOOLS",
    "SETTINGS_PROFILE_ENV",
    "WRITE_TOOLS",
    "Decision",
    "build_options",
    "decide",
    "setting_sources",
]

type Decision = Literal["allow", "deny"]

READ_TOOLS = frozenset({"Read", "Glob", "Grep"})
WRITE_TOOLS = frozenset({"Edit", "Write", "NotebookEdit"})
SETTINGS_PROFILE_ENV = "NEXUS_CLAUDE_SETTINGS_PROFILE"
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({"diff", "log", "show"})
_PLAIN_COMMAND = re.compile(r"[A-Za-z0-9 _./:=@~^,+-]+")
_HELPERS_OFF = frozenset({"--no-ext-diff", "--no-textconv"})
_HELPERS_ON = frozenset({"--ext-diff", "--textconv"})
_SETTING_SOURCES: dict[str, list[str] | None] = {
    "isolated": [],
    "project": ["project"],
    "inherit": None,
}
_WORKSPACE_SANDBOX: dict[str, bool] = {
    "enabled": True,
    "autoAllowBashIfSandboxed": True,
    "allowUnsandboxedCommands": False,
    "failIfUnavailable": True,
}
_DENY_MESSAGE = "Denied by nexus-mcp profile policy"


def _is_read_only_git(tool_input: Mapping[str, Any]) -> bool:
    command = tool_input.get("command")
    if not isinstance(command, str) or not _PLAIN_COMMAND.fullmatch(command):
        return False
    tokens = command.split()
    names = {token.split("=", 1)[0] for token in tokens}
    options = tokens[: tokens.index("--")] if "--" in tokens else tokens
    return (
        len(tokens) >= 2
        and tokens[0] == "git"
        and tokens[1] in _READ_ONLY_GIT_SUBCOMMANDS
        and not any(token.startswith("--output") for token in tokens)
        and set(options) >= _HELPERS_OFF
        and not (_HELPERS_ON & names)
    )


def _inside_workspace(tool: str, workspace: Path, tool_input: Mapping[str, Any]) -> bool:
    raw = tool_input.get("notebook_path" if tool == "NotebookEdit" else "file_path")
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        return False
    try:
        return (workspace / raw).resolve().is_relative_to(workspace.resolve())
    except (OSError, ValueError):
        return False


def decide(tool: str, tool_input: Mapping[str, Any], *, profile: Profile, cwd: Path) -> Decision:
    """Return the single authoritative permission decision for one Claude tool call."""
    if profile == "full_access" or tool in READ_TOOLS:
        return "allow"
    if tool == "Bash" and _is_read_only_git(tool_input):
        return "allow"
    if profile == "workspace_write":
        if tool == "Bash":
            return "allow"  # confined by the OS sandbox configured in build_options
        if tool in WRITE_TOOLS and _inside_workspace(tool, cwd, tool_input):
            return "allow"
    return "deny"


def setting_sources() -> list[str] | None:
    """Map NEXUS_CLAUDE_SETTINGS_PROFILE to the SDK's setting_sources."""
    profile = os.environ.get(SETTINGS_PROFILE_ENV, "isolated")
    if profile not in _SETTING_SOURCES:
        raise ToolError(f"{SETTINGS_PROFILE_ENV} must be one of isolated, project, inherit")
    return _SETTING_SOURCES[profile]


def _gate(profile: Profile, cwd: Path) -> HookCallback:
    async def gate(
        input_data: dict[str, Any], tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        del tool_use_id, context
        tool_input = input_data.get("tool_input")
        verdict = decide(
            str(input_data.get("tool_name") or ""),
            tool_input if isinstance(tool_input, dict) else {},
            profile=profile,
            cwd=cwd,
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": verdict,
                "permissionDecisionReason": "nexus-mcp profile policy",
            }
        }

    return cast("HookCallback", gate)


def _permission_gate(profile: Profile, cwd: Path) -> CanUseTool:
    async def check(
        tool: str, tool_input: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        del context
        if decide(tool, tool_input, profile=profile, cwd=cwd) == "allow":
            return PermissionResultAllow()
        return PermissionResultDeny(message=_DENY_MESSAGE)

    return check


def build_options(
    *, cwd: Path, profile: Profile, model: str | None, resume: str | None, fork: bool
) -> ClaudeAgentOptions:
    """Build fail-closed SDK options; decide() is the only permission authority."""
    return ClaudeAgentOptions(
        cwd=str(cwd),
        permission_mode="default",
        strict_mcp_config=True,
        mcp_servers={},
        disallowed_tools=["AskUserQuestion"],
        can_use_tool=_permission_gate(profile, cwd),
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_gate(profile, cwd)])]},
        setting_sources=cast("Any", setting_sources()),
        sandbox=(
            cast("SandboxSettings", dict(_WORKSPACE_SANDBOX))
            if profile == "workspace_write"
            else None
        ),
        verbatim_prompts=True,
        model=model,
        resume=resume,
        fork_session=fork,
    )
