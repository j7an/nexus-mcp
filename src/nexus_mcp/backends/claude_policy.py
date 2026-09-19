"""Pure permission policy and SDK option construction for the Claude Agent backend."""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from claude_agent_sdk import CanUseTool, ClaudeAgentOptions, HookCallback, HookMatcher

from nexus_mcp.config import ClaudeSettingsProfile
from nexus_mcp.core import ApprovalPolicy, SandboxMode

if TYPE_CHECKING:
    from claude_agent_sdk.types import SandboxSettings

__all__ = [
    "READ_TOOLS",
    "REVIEW_GIT_SUBCOMMANDS",
    "WRITE_TOOLS",
    "Decision",
    "build_options",
    "decide",
]

type Decision = Literal["allow", "ask", "deny"]

READ_TOOLS = frozenset({"Read", "Glob", "Grep"})
WRITE_TOOLS = frozenset({"Edit", "Write", "NotebookEdit"})
REVIEW_GIT_SUBCOMMANDS = frozenset({"diff", "log", "show"})
_PLAIN_COMMAND = re.compile(r"[A-Za-z0-9 _./:=@~^,+-]+")
_HELPERS_OFF = frozenset({"--no-ext-diff", "--no-textconv"})
_HELPERS_ON = frozenset({"--ext-diff", "--textconv"})
_SETTING_SOURCES: dict[str, list[str] | None] = {
    "isolated": [],
    "project": ["project"],
    "inherit": None,
}


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
        and tokens[1] in REVIEW_GIT_SUBCOMMANDS
        and not any(token.startswith("--output") for token in tokens)
        and set(options) >= _HELPERS_OFF
        and not (_HELPERS_ON & names)
    )


def _inside_workspace(workspace: Path, tool_input: Mapping[str, Any]) -> bool:
    raw = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not isinstance(raw, str) or not raw:
        return False
    return (workspace / raw).resolve().is_relative_to(workspace.resolve())


def decide(
    tool: str,
    tool_input: Mapping[str, Any],
    *,
    sandbox: SandboxMode,
    approval: ApprovalPolicy,
    workspace: Path,
    review: bool = False,
) -> Decision:
    """Return the single authoritative permission decision for one Claude tool call."""
    if sandbox == "danger_full_access" or tool in READ_TOOLS:
        return "allow"
    if review and tool == "Bash" and _is_read_only_git(tool_input):
        return "allow"
    if sandbox == "workspace_write":
        if tool == "Bash":
            return "allow"
        if tool in WRITE_TOOLS and _inside_workspace(workspace, tool_input):
            return "allow"
        return "deny" if approval == "never" else "ask"
    return "deny"


def build_options(
    *,
    workspace: Path,
    sandbox: SandboxMode,
    profile: ClaudeSettingsProfile,
    can_use_tool: CanUseTool,
    pre_tool_use: HookCallback,
    resume: str | None = None,
    model: str | None = None,
    output_schema: Mapping[str, Any] | None = None,
    cli_path: str | None = None,
) -> ClaudeAgentOptions:
    """Build fail-closed SDK options; decide() stays the only permission authority."""
    sandbox_settings = (
        cast(
            "SandboxSettings",
            {
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "failIfUnavailable": True,
            },
        )
        if sandbox == "workspace_write"
        else None
    )
    return ClaudeAgentOptions(
        cwd=str(workspace),
        permission_mode="default",
        strict_mcp_config=True,
        mcp_servers={},
        env={},
        allowed_tools=sorted(READ_TOOLS),
        disallowed_tools=["AskUserQuestion"],
        can_use_tool=can_use_tool,
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[pre_tool_use])]},
        setting_sources=cast("Any", _SETTING_SOURCES[profile]),
        sandbox=sandbox_settings,
        resume=resume,
        model=model,
        cli_path=cli_path,
        output_format=(
            None
            if output_schema is None
            else {"type": "json_schema", "schema": dict(output_schema)}
        ),
    )
