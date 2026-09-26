"""Opt-in smoke coverage for the real Codex App Server backend."""

import json
import re
from pathlib import Path

import pytest

from nexus_mcp.backends import codex
from nexus_mcp.types import PromptRequest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_CALL_TYPES = frozenset({"function_call", "custom_tool_call"})
_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})
_EXECUTION_NAMES = frozenset({"exec", "exec_command"})
_NONZERO_EXIT = re.compile(
    r"(?:exit_code|exit code)\D*(-?[1-9]\d*)|exited with code\s+([1-9]\d*)", re.I
)
_PERMISSION_DENIAL = re.compile(r"operation not permitted|permission denied", re.I)


def _matching_execution_outputs(rollout_path: str, command: str) -> list[str]:
    """Return only runtime outputs whose call ID invoked the target command."""
    records = [json.loads(line) for line in Path(rollout_path).read_text().splitlines()]
    return _correlated_execution_outputs(records, command)


def _correlated_execution_outputs(records: list[dict[str, object]], command: str) -> list[str]:
    """Correlate native runtime calls and outputs by call ID."""
    call_ids = {
        payload["call_id"]
        for record in records
        if record.get("type") == "response_item"
        and isinstance(payload := record.get("payload"), dict)
        and payload.get("type") in _CALL_TYPES
        and payload.get("name") in _EXECUTION_NAMES
        and isinstance(payload.get("call_id"), str)
        and command in json.dumps(payload.get("input", payload.get("arguments")))
    }
    return [
        json.dumps(payload.get("output"))
        for record in records
        if record.get("type") == "response_item"
        and isinstance(payload := record.get("payload"), dict)
        and payload.get("type") in _OUTPUT_TYPES
        and payload.get("call_id") in call_ids
    ]


def _is_permission_denial_with_nonzero_exit(output: str) -> bool:
    return bool(_NONZERO_EXIT.search(output) and _PERMISSION_DENIAL.search(output))


async def test_new_continue_fork(codex_installed: None, tmp_path):
    """The bundled CLI shares Codex auth and works with experimental API disabled."""
    first = await codex.run(
        PromptRequest(prompt="Reply with exactly the word: alpha", cwd=tmp_path)
    )
    assert "alpha" in first.output.lower()
    again = await codex.run(
        PromptRequest(
            prompt="Which word did you reply with? One word.",
            cwd=tmp_path,
            session_id=first.session_id,
        )
    )
    assert again.session_id == first.session_id
    assert "alpha" in again.output.lower()
    forked = await codex.run(
        PromptRequest(
            prompt="Reply with: beta", cwd=tmp_path, session_id=first.session_id, fork=True
        )
    )
    assert forked.session_id != first.session_id


async def test_read_only_continuation_narrows_full_access_thread(codex_installed: None, tmp_path):
    """A continued read-only thread records a denied target-command execution."""
    opened = await codex.run(
        PromptRequest(prompt="Reply with: ready", cwd=tmp_path, profile="full_access")
    )
    await codex.run(
        PromptRequest(
            prompt="Run exactly this shell command: touch narrowed.txt . Then stop.",
            cwd=tmp_path,
            profile="read_only",
            session_id=opened.session_id,
        )
    )
    async with codex._session() as client:
        resumed = await client.thread_resume(opened.session_id)
        history = await resumed.read(include_turns=True)
    assert history.thread.path is not None
    outputs = _matching_execution_outputs(history.thread.path, "touch narrowed.txt")
    assert outputs, "no matching runtime command output was recorded"
    assert any(_is_permission_denial_with_nonzero_exit(output) for output in outputs)
    assert not (tmp_path / "narrowed.txt").exists()


async def test_workspace_write_allows_writes_inside_cwd(codex_installed: None, tmp_path):
    """The workspace-write profile permits a write inside its current directory."""
    await codex.run(
        PromptRequest(
            prompt="Create a file allowed.txt containing hi. Then stop.",
            cwd=tmp_path,
            profile="workspace_write",
        )
    )
    assert (tmp_path / "allowed.txt").read_text().strip() == "hi"


async def test_models_listed(codex_installed: None):
    """The App Server exposes at least one visible model."""
    listed = await codex.info()
    assert listed.models
