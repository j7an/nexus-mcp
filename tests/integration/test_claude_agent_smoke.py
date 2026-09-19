"""Opt-in smoke test against the real Claude Agent SDK. Consumes provider usage."""

from contextlib import suppress

import pytest

from nexus_mcp.backends.base import BackendFailure
from nexus_mcp.backends.claude_agent import ClaudeAgentBackend
from nexus_mcp.core import ForkOperation, ProviderReference, ResolvedExecutionConfig, TurnOperation
from tests.unit.backends.claude_fakes import FakeContext

pytestmark = pytest.mark.integration


def _context(path, *, checkpoint: tuple[ProviderReference, ...] = (), **config):
    return FakeContext(
        workspace_path=path,
        resolved_config=ResolvedExecutionConfig(model="haiku", **config),
        source_checkpoint=checkpoint,
    )


async def test_turn_resume_fork_structured(tmp_path):
    """A real SDK session resumes, forks, and returns a structured response."""
    backend = ClaudeAgentBackend()
    first = _context(tmp_path)
    await backend.execute(TurnOperation(prompt="Remember the word 'kiwi'. Reply OK."), first)
    [session] = first.references

    second = _context(tmp_path, checkpoint=(session,))
    outcome = await backend.execute(
        TurnOperation(
            prompt="What word did I ask you to remember?",
            output_schema={
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"],
            },
        ),
        second,
    )
    assert second.references == [session]
    assert "kiwi" in outcome.structured_output["word"].lower()

    third = _context(tmp_path, checkpoint=(session,))
    third.session = third.session.model_copy(update={"parent_session_id": "session-parent"})
    forked = await backend.execute(ForkOperation(prompt="Reply OK."), third)
    assert forked.provider_reference != session


async def test_workspace_write_contains_writes(tmp_path):
    """The workspace-write sandbox permits writes only inside the workspace."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    context = _context(workspace, sandbox="workspace_write", approval_policy="never")
    await ClaudeAgentBackend().execute(
        TurnOperation(
            prompt=(
                "Use the Write tool to create inside.txt containing 'a', then use the "
                f"Write tool to create {outside} containing 'b'. Do not use Bash."
            )
        ),
        context,
    )
    assert (workspace / "inside.txt").exists()
    assert not outside.exists()


async def test_sandbox_failure_is_closed(tmp_path, monkeypatch):
    """failIfUnavailable stops the turn when the OS sandbox cannot start."""
    monkeypatch.setenv("PATH", str(tmp_path))
    context = _context(tmp_path, sandbox="workspace_write", approval_policy="never")
    marker = tmp_path / "ran.txt"
    with suppress(BackendFailure):
        await ClaudeAgentBackend().execute(
            TurnOperation(prompt=f"Run this exact Bash command: touch {marker}"), context
        )
    assert not marker.exists(), "Bash ran unsandboxed: failIfUnavailable is not enforced"
