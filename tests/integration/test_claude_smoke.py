"""Opt-in smoke coverage for the real Claude CLI backend."""

import pytest

from nexus_mcp.backends import claude, claude_policy
from nexus_mcp.backends.claude_policy import SETTINGS_PROFILE_ENV
from nexus_mcp.types import PromptRequest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MODEL = "haiku"


async def test_new_continue_fork(claude_installed: None, tmp_path):
    """A new Claude session can continue and fork."""
    first = await claude.run(
        PromptRequest(prompt="Reply with exactly the word: alpha", cwd=tmp_path, model=MODEL)
    )
    assert "alpha" in first.output.lower()
    again = await claude.run(
        PromptRequest(
            prompt="Which word did you reply with? One word.",
            cwd=tmp_path,
            model=MODEL,
            session_id=first.session_id,
        )
    )
    assert again.session_id == first.session_id
    assert "alpha" in again.output.lower()
    forked = await claude.run(
        PromptRequest(
            prompt="Reply with exactly the word: beta",
            cwd=tmp_path,
            model=MODEL,
            session_id=first.session_id,
            fork=True,
        )
    )
    assert forked.session_id != first.session_id


@pytest.mark.parametrize("settings_profile", ["isolated", "inherit"])
async def test_hook_denies_attempted_write_despite_allow_rule(
    claude_installed: None, tmp_path, monkeypatch, settings_profile: str
):
    """The PreToolUse deny must win over a permissive allow rule.

    The allow rule is supplied through `allowed_tools`, which the SDK documents as
    equivalent to a settings allow rule. Project settings files cannot serve as the
    controlled rule: the CLI ignores them in an untrusted workspace such as tmp_path
    (live probe recorded in the 2026-09-19 Claude backend spec).
    """
    monkeypatch.setenv(SETTINGS_PROFILE_ENV, settings_profile)
    decisions: list[tuple[str, str]] = []
    real_decide = claude_policy.decide

    def recording_decide(tool, tool_input, **kwargs):
        verdict = real_decide(tool, tool_input, **kwargs)
        decisions.append((tool, verdict))
        return verdict

    real_build = claude.build_options

    def build_with_allow_rule(**kwargs):
        options = real_build(**kwargs)
        options.allowed_tools = ["Write"]
        return options

    monkeypatch.setattr(claude_policy, "decide", recording_decide)
    monkeypatch.setattr(claude, "build_options", build_with_allow_rule)
    await claude.run(
        PromptRequest(
            prompt="Use the Write tool to create denied.txt containing hi. Then stop.",
            cwd=tmp_path,
            model=MODEL,
        )
    )
    assert ("Write", "deny") in decisions
    assert not (tmp_path / "denied.txt").exists()


async def test_workspace_write_allows_writes_inside_cwd(claude_installed: None, tmp_path):
    """The workspace-write profile permits a write inside its current directory."""
    await claude.run(
        PromptRequest(
            prompt="Use the Write tool to create allowed.txt containing hi. Then stop.",
            cwd=tmp_path,
            model=MODEL,
            profile="workspace_write",
        )
    )
    assert (tmp_path / "allowed.txt").read_text().strip() == "hi"
