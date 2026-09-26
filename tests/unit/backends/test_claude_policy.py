from pathlib import Path

import pytest
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext
from fastmcp.exceptions import ToolError

from nexus_mcp.backends.claude_policy import (
    SETTINGS_PROFILE_ENV,
    build_options,
    decide,
    setting_sources,
)

SAFE = "--no-ext-diff --no-textconv"


@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("Read", "allow"),
        ("Glob", "allow"),
        ("Grep", "allow"),
        ("Write", "deny"),
        ("Edit", "deny"),
        ("Bash", "deny"),
        ("WebFetch", "deny"),
        ("Task", "deny"),
        ("TodoWrite", "deny"),
        ("Unknown", "deny"),
    ],
)
def test_read_only_table(tmp_path, tool, expected):
    tool_input = {"file_path": str(tmp_path / "a.txt"), "command": "ls"}
    assert decide(tool, tool_input, profile="read_only", cwd=tmp_path) == expected


@pytest.mark.parametrize("tool", ["Read", "Write", "Bash", "WebFetch", "Unknown"])
def test_full_access_allows_everything(tmp_path, tool):
    assert decide(tool, {}, profile="full_access", cwd=tmp_path) == "allow"


def test_workspace_write_table(tmp_path):
    def run(tool, tool_input):
        return decide(tool, tool_input, profile="workspace_write", cwd=tmp_path)

    assert run("Read", {}) == "allow"
    assert run("Bash", {"command": "ls"}) == "allow"
    assert run("Write", {"file_path": str(tmp_path / "in.txt")}) == "allow"
    assert run("Edit", {"file_path": "relative/in.txt"}) == "allow"
    assert run("NotebookEdit", {"notebook_path": str(tmp_path / "n.ipynb")}) == "allow"
    assert run("WebFetch", {"url": "https://example.com"}) == "deny"
    assert run("TotallyUnknownTool", {}) == "deny"


@pytest.mark.parametrize(
    "tool_input",
    [
        {"file_path": "../escape.txt"},
        {"file_path": "/etc/hosts"},
        {"file_path": "link/escape.txt"},
        {},
        {"file_path": 7},
        {"file_path": ""},
        {"file_path": "a\x00b"},
    ],
)
def test_workspace_write_denies_paths_outside_cwd(tmp_path, tool_input):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    assert decide("Write", tool_input, profile="workspace_write", cwd=workspace) == "deny"


def test_workspace_write_rejects_nul_even_if_resolve_accepts_it(monkeypatch):
    """Windows can resolve a NUL-bearing path as an inside-workspace path."""
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    decision = decide(
        "Write", {"file_path": "a\x00b"}, profile="workspace_write", cwd=Path("/workspace")
    )
    assert decision == "deny"


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_workspace_write_denies_path_when_resolution_fails(monkeypatch, error):
    def cannot_resolve(_path):
        raise error("bad path")

    monkeypatch.setattr(Path, "resolve", cannot_resolve)
    decision = decide(
        "Write", {"file_path": "inside.txt"}, profile="workspace_write", cwd=Path("/workspace")
    )
    assert decision == "deny"


def test_notebook_edit_uses_notebook_path_for_containment(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tool_input = {"file_path": str(workspace / "in.txt"), "notebook_path": "../out.ipynb"}
    assert decide("NotebookEdit", tool_input, profile="workspace_write", cwd=workspace) == "deny"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (f"git diff {SAFE}", "allow"),
        (f"git diff {SAFE} main...HEAD -- src/a.py", "allow"),
        (f"git log {SAFE} -n 5 --stat", "allow"),
        (f"git show {SAFE} HEAD~1:README.md", "allow"),
        ("git diff", "deny"),
        ("git diff --no-ext-diff", "deny"),
        ("git diff --no-textconv", "deny"),
        ("git diff -- src --no-ext-diff --no-textconv", "deny"),
        ("git diff --no-ext-diff -- src --no-textconv", "deny"),
        (f"git diff {SAFE} --ext-diff", "deny"),
        (f"git diff {SAFE} --textconv", "deny"),
        (f"git log --ext-diff {SAFE} -p", "deny"),
        (f"git diff {SAFE} --output=/tmp/x", "deny"),
        (f"git log {SAFE} --output /tmp/x", "deny"),
        (f"git diff {SAFE} && rm -rf x", "deny"),
        (f"git diff {SAFE}; touch x", "deny"),
        (f"git diff {SAFE} > x", "deny"),
        (f"git diff {SAFE} $(touch x)", "deny"),
        (f"git diff {SAFE} 'a b'", "deny"),
        (f"git -c core.pager=x diff {SAFE}", "deny"),
        ("git push", "deny"),
        ("ls", "deny"),
        ("git", "deny"),
        (7, "deny"),
    ],
)
def test_read_only_allows_only_read_only_git(tmp_path, command, expected):
    assert decide("Bash", {"command": command}, profile="read_only", cwd=tmp_path) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, []), ("isolated", []), ("project", ["project"]), ("inherit", None)],
)
def test_setting_sources(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(SETTINGS_PROFILE_ENV, raising=False)
    else:
        monkeypatch.setenv(SETTINGS_PROFILE_ENV, value)
    assert setting_sources() == expected


def test_setting_sources_rejects_unknown(monkeypatch):
    monkeypatch.setenv(SETTINGS_PROFILE_ENV, "everything")
    with pytest.raises(ToolError, match=SETTINGS_PROFILE_ENV):
        setting_sources()


def _options(tmp_path, **overrides):
    arguments = {
        "cwd": tmp_path,
        "profile": "read_only",
        "model": None,
        "resume": None,
        "fork": False,
    } | overrides
    return build_options(**arguments)


@pytest.mark.parametrize("profile", ["read_only", "workspace_write", "full_access"])
def test_options_invariants(monkeypatch, tmp_path, profile):
    monkeypatch.delenv(SETTINGS_PROFILE_ENV, raising=False)
    options = _options(tmp_path, profile=profile)
    assert options.cwd == str(tmp_path)
    assert options.permission_mode == "default"
    assert options.strict_mcp_config is True
    assert options.mcp_servers == {}
    assert options.disallowed_tools == ["AskUserQuestion"]
    assert options.verbatim_prompts is True
    assert options.setting_sources == []
    [matcher] = options.hooks["PreToolUse"]
    assert matcher.matcher is None
    assert len(matcher.hooks) == 1


def test_options_sandbox_only_for_workspace_write(tmp_path):
    assert _options(tmp_path, profile="read_only").sandbox is None
    assert _options(tmp_path, profile="full_access").sandbox is None
    assert _options(tmp_path, profile="workspace_write").sandbox == {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
        "failIfUnavailable": True,
    }


def test_options_session_and_model_fields(tmp_path):
    bare = _options(tmp_path)
    assert bare.resume is None and bare.model is None and bare.fork_session is False
    full = _options(tmp_path, resume="sid", model="haiku", fork=True)
    assert full.resume == "sid" and full.model == "haiku" and full.fork_session is True


async def test_hook_returns_explicit_allow_and_deny(tmp_path):
    [matcher] = _options(tmp_path, profile="workspace_write").hooks["PreToolUse"]
    gate = matcher.hooks[0]
    inside = {"tool_name": "Write", "tool_input": {"file_path": str(tmp_path / "a.txt")}}
    outside = {"tool_name": "Write", "tool_input": {"file_path": "/etc/hosts"}}
    allowed = await gate(inside, "tool-1", None)
    denied = await gate(outside, "tool-2", None)
    assert allowed["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert allowed["hookSpecificOutput"]["hookEventName"] == "PreToolUse"


async def test_hook_denies_malformed_input(tmp_path):
    [matcher] = _options(tmp_path).hooks["PreToolUse"]
    result = await matcher.hooks[0]({"tool_name": None, "tool_input": "nope"}, None, None)
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    ("profile", "tool", "tool_input", "expected"),
    [
        ("read_only", "Read", {}, PermissionResultAllow),
        ("read_only", "Write", {"file_path": ".vscode/settings.json"}, PermissionResultDeny),
        ("read_only", "Bash", {"command": f"git diff {SAFE}"}, PermissionResultAllow),
        ("read_only", "Bash", {"command": "touch forbidden.txt"}, PermissionResultDeny),
        ("workspace_write", "Write", {"file_path": ".vscode/settings.json"}, PermissionResultAllow),
        (
            "workspace_write",
            "Edit",
            {"file_path": ".pre-commit-config.yaml"},
            PermissionResultAllow,
        ),
        ("workspace_write", "Write", {"file_path": "../outside.txt"}, PermissionResultDeny),
        ("workspace_write", "WebFetch", {"url": "https://example.com"}, PermissionResultDeny),
        ("full_access", "Write", {"file_path": "/etc/hosts"}, PermissionResultAllow),
        ("full_access", "Bash", {"command": "touch allowed.txt"}, PermissionResultAllow),
    ],
)
async def test_can_use_tool_matches_profile_decision(tmp_path, profile, tool, tool_input, expected):
    options = _options(tmp_path, profile=profile)
    result = await options.can_use_tool(tool, tool_input, ToolPermissionContext())
    assert isinstance(result, expected)
    assert result.behavior == decide(tool, tool_input, profile=profile, cwd=tmp_path)
