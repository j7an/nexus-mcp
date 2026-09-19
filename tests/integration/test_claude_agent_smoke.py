"""Opt-in smoke test against the real Claude Agent SDK. Consumes provider usage."""

from collections.abc import Iterable
from shutil import which
from typing import Any

import pytest
from claude_agent_sdk import AssistantMessage, ToolResultBlock, ToolUseBlock, UserMessage

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


def _tool_blocks(messages: Iterable[object]) -> Iterable[ToolUseBlock | ToolResultBlock]:
    """Yield raw SDK tool blocks for smoke-test evidence only."""
    for message in messages:
        if isinstance(message, (AssistantMessage, UserMessage)) and isinstance(
            message.content, list
        ):
            yield from (
                block
                for block in message.content
                if isinstance(block, (ToolUseBlock, ToolResultBlock))
            )


def _tool_result_text(content: str | list[dict[str, Any]] | None) -> str:
    """Extract textual diagnostics from either SDK tool-result content shape."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""


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
    """A missing OS sandbox fails after the requested Bash command reaches the SDK."""
    touch = which("touch")
    assert touch is not None

    monkeypatch.setenv("PATH", str(tmp_path))
    context = _context(tmp_path, sandbox="workspace_write", approval_policy="never")
    marker = tmp_path / "ran.txt"
    command = f"{touch} {marker}"
    received: list[object] = []
    backend = ClaudeAgentBackend()
    translate = backend._translate

    async def record_raw_message(
        message: Any, execution_context: Any, recorded: set[str], pending: dict[str, str]
    ) -> Any:
        received.append(message)
        return await translate(message, execution_context, recorded, pending)

    monkeypatch.setattr(backend, "_translate", record_raw_message)
    try:
        await backend.execute(
            TurnOperation(prompt=f"Run this exact Bash command: {command}"), context
        )
    except BackendFailure as error:
        assert error.error.code == "provider_failed"

    blocks = tuple(_tool_blocks(received))
    calls = tuple(
        block for block in blocks if isinstance(block, ToolUseBlock) and block.name == "Bash"
    )
    assert any(call.input.get("command") == command for call in calls)
    assert any(event.type == "command" for event in context.events)

    call_ids = {call.id for call in calls if call.input.get("command") == command}
    failures = (
        block
        for block in blocks
        if isinstance(block, ToolResultBlock)
        and block.tool_use_id in call_ids
        and block.is_error is True
    )
    assert any(
        "sandbox" in _tool_result_text(result.content).casefold()
        and (
            "unavailable" in _tool_result_text(result.content).casefold()
            or "failed" in _tool_result_text(result.content).casefold()
        )
        for result in failures
    ), "Bash did not report sandbox unavailability"
    assert not marker.exists(), "Bash ran unsandboxed: failIfUnavailable is not enforced"
