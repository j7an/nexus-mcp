"""Antigravity SDK configuration contracts."""

from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

pytest.importorskip("google.antigravity")

from google.antigravity.hooks import policy
from google.antigravity.types import BuiltinTools, SessionContinuationMode

from nexus_mcp.backends import antigravity
from nexus_mcp.types import PromptRequest

CID = "0f9d3c1e-1111-4222-8333-444455556666"


@pytest.fixture(autouse=True)
def session_store(monkeypatch, tmp_path):
    monkeypatch.setattr(antigravity, "_STORE", tmp_path / "store")


def request(tmp_path: Path, **overrides) -> PromptRequest:
    return PromptRequest(prompt="do it", cwd=tmp_path, **overrides)


@pytest.mark.parametrize(
    ("profile", "tools", "sandbox"),
    [
        (
            "read_only",
            {"view_file", "list_directory", "search_directory", "find_file", "finish"},
            None,
        ),
        (
            "workspace_write",
            {
                "view_file",
                "list_directory",
                "search_directory",
                "find_file",
                "finish",
                "create_file",
                "edit_file",
                "run_command",
            },
            True,
        ),
    ],
)
def test_confined_profiles(tmp_path, profile, tools, sandbox):
    cfg = antigravity._config(request(tmp_path, profile=profile), CID)
    assert {t.value for t in cfg.capabilities.enabled_tools} == tools
    run_cfg = cfg.capabilities.run_command_config
    assert (run_cfg.enable_sandbox if run_cfg else None) == sandbox
    names = [p.name for p in cfg.policies]
    assert names[0] == "deny_all" and "allow_all" not in names
    assert "workspace_only" in names
    allowed = {p.tool for p in cfg.policies if p.decision == policy.Decision.APPROVE}
    assert allowed == tools


def test_full_access_profile(tmp_path):
    cfg = antigravity._config(request(tmp_path, profile="full_access"), CID)
    assert {t.value for t in cfg.capabilities.enabled_tools} == (
        {t.value for t in BuiltinTools} - {"ask_question"}
    )
    assert [p.name for p in cfg.policies] == ["allow_all"]


def test_new_session_fields(tmp_path):
    cfg = antigravity._config(request(tmp_path, model="gemini-x"), CID)
    assert cfg.conversation_id == CID
    assert cfg.session_continuation_mode == SessionContinuationMode.CREATE_ONLY
    assert cfg.save_dir == str(antigravity._STORE)
    assert cfg.workspaces == [str(tmp_path.resolve())]
    assert cfg.model == "gemini-x"


def test_continuation_uses_requested_profile(tmp_path):
    cfg = antigravity._config(request(tmp_path, session_id=CID, profile="read_only"), CID)
    assert cfg.session_continuation_mode == SessionContinuationMode.RESUME
    assert "allow_all" not in [p.name for p in cfg.policies]


def test_short_session_id_is_tool_error(tmp_path):
    with pytest.raises(ToolError, match="^session_id is not a valid Antigravity conversation id$"):
        antigravity._config(request(tmp_path, session_id="abc"), "abc")
