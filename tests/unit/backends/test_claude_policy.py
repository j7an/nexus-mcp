from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import PermissionResultAllow, ToolPermissionContext

from nexus_mcp.backends.claude_agent import ClaudeAgentBackend
from nexus_mcp.backends.claude_policy import READ_TOOLS, build_options, decide
from nexus_mcp.exceptions import ConfigurationError

APPROVALS = ["on_request", "provider_default", "never"]
SAFE = "--no-ext-diff --no-textconv"


async def _never_called(*_args):  # pragma: no cover - placeholder callback
    raise AssertionError


@pytest.mark.parametrize("approval", APPROVALS)
@pytest.mark.parametrize(
    ("tool", "expected"),
    [
        ("Read", "allow"),
        ("Write", "deny"),
        ("Bash", "deny"),
        ("WebFetch", "deny"),
        ("Unknown", "deny"),
    ],
)
def test_read_only_never_asks(tmp_path, tool, expected, approval):
    tool_input = {"file_path": str(tmp_path / "a.txt")}
    assert (
        decide(tool, tool_input, sandbox="read_only", approval=approval, workspace=tmp_path)
        == expected
    )


@pytest.mark.parametrize("review", [False, True])
def test_structured_output_is_allowed_for_read_only_and_review_turns(tmp_path, review):
    """A missing StructuredOutput exception blocks schema turns before output is returned."""
    assert (
        decide(
            "StructuredOutput",
            {},
            sandbox="read_only",
            approval="never",
            workspace=tmp_path,
            review=review,
        )
        == "allow"
    )


async def test_structured_output_policy_allows_sdk_permission_callbacks(tmp_path):
    context = SimpleNamespace(workspace=SimpleNamespace(canonical_path=tmp_path))
    backend = object.__new__(ClaudeAgentBackend)
    can_use_tool = backend._permission_handler(context, "read_only", "never", review=True)
    pre_tool_use = ClaudeAgentBackend._pre_tool_use(context, "read_only", "never", review=True)

    result = await can_use_tool("StructuredOutput", {}, ToolPermissionContext())
    gate = await pre_tool_use({"tool_name": "StructuredOutput", "tool_input": {}}, "tool-use", None)

    assert isinstance(result, PermissionResultAllow)
    assert gate == {}


@pytest.mark.parametrize("approval", APPROVALS)
@pytest.mark.parametrize("tool", ["Read", "Write", "Bash", "WebFetch", "Unknown"])
def test_danger_full_access_allows_everything(tmp_path, tool, approval):
    assert (
        decide(tool, {}, sandbox="danger_full_access", approval=approval, workspace=tmp_path)
        == "allow"
    )


@pytest.mark.parametrize(
    ("approval", "escalated"),
    [("on_request", "ask"), ("provider_default", "ask"), ("never", "deny")],
)
def test_workspace_write_table(tmp_path, approval, escalated):
    def run(tool, tool_input):
        return decide(
            tool, tool_input, sandbox="workspace_write", approval=approval, workspace=tmp_path
        )

    assert run("Read", {}) == "allow"
    assert run("Bash", {"command": "ls"}) == "allow"
    assert run("Write", {"file_path": str(tmp_path / "in.txt")}) == "allow"
    assert run("Edit", {"file_path": "relative/in.txt"}) == "allow"
    assert run("NotebookEdit", {"notebook_path": str(tmp_path / "n.ipynb")}) == "allow"
    assert run("WebFetch", {"url": "https://example.com"}) == escalated
    assert run("TotallyUnknownTool", {}) == escalated


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
def test_workspace_write_escalates_paths_outside_workspace(tmp_path, tool_input):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    assert (
        decide(
            "Write", tool_input, sandbox="workspace_write", approval="never", workspace=workspace
        )
        == "deny"
    )


def test_workspace_write_rejects_nul_even_if_resolve_accepts_it(monkeypatch):
    """Windows can resolve a NUL-bearing path as an inside-workspace path."""
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    assert (
        decide(
            "Write",
            {"file_path": "a\x00b"},
            sandbox="workspace_write",
            approval="never",
            workspace=Path("/workspace"),
        )
        == "deny"
    )


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_workspace_write_denies_path_when_resolution_fails(monkeypatch, error):
    def cannot_resolve(_path):
        raise error("bad path")

    monkeypatch.setattr(Path, "resolve", cannot_resolve)
    assert (
        decide(
            "Write",
            {"file_path": "inside.txt"},
            sandbox="workspace_write",
            approval="never",
            workspace=Path("/workspace"),
        )
        == "deny"
    )


def test_notebook_edit_uses_notebook_path_for_workspace_containment(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert (
        decide(
            "NotebookEdit",
            {"file_path": str(workspace / "inside.txt"), "notebook_path": "../outside.ipynb"},
            sandbox="workspace_write",
            approval="never",
            workspace=workspace,
        )
        == "deny"
    )


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
def test_review_allows_only_read_only_git(tmp_path, command, expected):
    assert (
        decide(
            "Bash",
            {"command": command},
            sandbox="read_only",
            approval="never",
            workspace=tmp_path,
            review=True,
        )
        == expected
    )


def test_git_is_denied_outside_review(tmp_path):
    assert (
        decide(
            "Bash",
            {"command": f"git diff {SAFE}"},
            sandbox="read_only",
            approval="never",
            workspace=tmp_path,
        )
        == "deny"
    )


def _options(tmp_path: Path, **overrides):
    arguments = {
        "workspace": tmp_path,
        "sandbox": "read_only",
        "profile": "isolated",
        "can_use_tool": _never_called,
        "pre_tool_use": _never_called,
    } | overrides
    return build_options(**arguments)


@pytest.mark.parametrize("sandbox", ["read_only", "workspace_write", "danger_full_access"])
def test_options_invariants(tmp_path, sandbox):
    options = _options(tmp_path, sandbox=sandbox)
    assert options.cwd == str(tmp_path)
    assert options.permission_mode == "default"
    assert options.strict_mcp_config is True
    assert options.mcp_servers == {}
    assert options.env == {}
    assert set(options.allowed_tools) == READ_TOOLS
    assert options.disallowed_tools == ["AskUserQuestion"]
    assert options.can_use_tool is _never_called
    [matcher] = options.hooks["PreToolUse"]
    assert matcher.matcher is None
    assert matcher.hooks == [_never_called]


def test_options_sandbox_only_for_workspace_write(tmp_path):
    assert _options(tmp_path, sandbox="read_only").sandbox is None
    assert _options(tmp_path, sandbox="danger_full_access").sandbox is None
    assert _options(tmp_path, sandbox="workspace_write").sandbox == {
        "enabled": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
        "failIfUnavailable": True,
    }


@pytest.mark.parametrize(
    ("profile", "expected"), [("isolated", []), ("project", ["project"]), ("inherit", None)]
)
def test_options_profile(tmp_path, profile, expected):
    assert _options(tmp_path, profile=profile).setting_sources == expected


def test_options_rejects_unknown_profile(tmp_path):
    with pytest.raises(ConfigurationError):
        _options(tmp_path, profile="invalid")


def test_options_optional_fields(tmp_path):
    bare = _options(tmp_path)
    assert bare.resume is None and bare.model is None and bare.output_format is None
    assert bare.cli_path is None and bare.fork_session is False
    schema = {"type": "object"}
    full = _options(
        tmp_path, resume="sid", model="haiku", output_schema=schema, cli_path="/x/claude"
    )
    assert full.resume == "sid" and full.model == "haiku" and full.cli_path == "/x/claude"
    assert full.output_format == {"type": "json_schema", "schema": schema}
