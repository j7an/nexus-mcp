"""Docker-only smoke coverage for the real Antigravity harness."""

import asyncio
import os
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("google.antigravity")

from fastmcp.exceptions import ToolError

from nexus_mcp import server
from nexus_mcp.backends import antigravity
from nexus_mcp.types import PromptRequest

pytestmark = pytest.mark.integration

_START_ERROR = (
    "Antigravity could not start or resume the session "
    "(unknown session_id, or the harness failed to start)"
)
_CONNECTION_ERROR = (
    "Antigravity run failed (AntigravityConnectionError); check credentials and model"
)


@pytest.fixture(autouse=True)
def isolated_store(antigravity_installed: None, tmp_path: Path, monkeypatch):
    """Keep all smoke-test conversation storage in the temporary directory."""
    monkeypatch.setattr(antigravity, "_STORE", tmp_path / "store")


@pytest.fixture
def invalid_key(monkeypatch):
    """Use a deliberately invalid key without consulting host credentials."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")


@pytest.fixture
def gemini_key(monkeypatch):
    """Enable authenticated tests only through the explicit smoke-test key."""
    key = os.environ.get("NEXUS_TEST_GEMINI_API_KEY")
    if not key:
        pytest.skip("NEXUS_TEST_GEMINI_API_KEY not set")
    monkeypatch.setenv("GEMINI_API_KEY", key)


async def test_unknown_session_is_tool_error(invalid_key: None, tmp_path: Path):
    """An unknown conversation fails with the fixed startup error."""
    with pytest.raises(ToolError, match=f"^{re.escape(_START_ERROR)}$"):
        await antigravity.run(
            PromptRequest(prompt="Reply OK.", cwd=tmp_path, session_id=str(uuid4()))
        )


async def test_invalid_key_is_tool_error(invalid_key: None, tmp_path: Path):
    """A rejected key produces a connection error without provider free text."""
    with pytest.raises(ToolError, match=f"^{re.escape(_CONNECTION_ERROR)}$") as error:
        await antigravity.run(PromptRequest(prompt="Reply OK.", cwd=tmp_path))
    assert "API key not valid" not in str(error.value)


async def test_workspace_write_passes_sandbox_gate_in_container(invalid_key: None, tmp_path: Path):
    """The container sandbox permits startup to reach the connection failure."""
    with pytest.raises(ToolError, match=f"^{re.escape(_CONNECTION_ERROR)}$"):
        await antigravity.run(
            PromptRequest(prompt="Reply OK.", cwd=tmp_path, profile="workspace_write")
        )


async def test_new_then_continue_recalls(gemini_key: None, tmp_path: Path):
    """A continued conversation recalls the first turn and retains its ID."""
    first = await antigravity.run(
        PromptRequest(prompt="Remember the word PELICAN. Reply OK.", cwd=tmp_path)
    )
    continued = await antigravity.run(
        PromptRequest(
            prompt="What word did I ask you to remember?",
            cwd=tmp_path,
            session_id=first.session_id,
        )
    )
    assert continued.session_id == first.session_id
    assert "PELICAN" in continued.output.upper()


async def test_read_only_cannot_write(gemini_key: None, tmp_path: Path):
    """The read-only tool set prevents creating a file in the workspace."""
    await antigravity.run(
        PromptRequest(
            prompt="Create probe.txt containing exactly ok in the current directory. Then stop.",
            cwd=tmp_path,
            profile="read_only",
        )
    )
    assert not (tmp_path / "probe.txt").exists()


async def test_workspace_write_writes_inside_cwd(gemini_key: None, tmp_path: Path):
    """Workspace-write permits creating the requested file inside cwd."""
    await antigravity.run(
        PromptRequest(
            prompt="Create probe.txt containing exactly ok in the current directory. Then stop.",
            cwd=tmp_path,
            profile="workspace_write",
        )
    )
    assert (tmp_path / "probe.txt").read_text().strip() == "ok"


async def test_timeout_leaves_no_harness_process(gemini_key: None, tmp_path: Path):
    """A timed-out real turn tears down the harness before the cleanup deadline."""
    with pytest.raises(ToolError, match="timed out"):
        await server.run_prompt(
            backend="antigravity",
            prompt="Keep working for thirty seconds before replying. Explain Python fully.",
            cwd=str(tmp_path),
            timeout=1,
        )
    deadline = time.monotonic() + 5
    while True:
        observed = subprocess.run(["pgrep", "-f", "[l]ocalharness"], capture_output=True, text=True)
        assert observed.returncode in (0, 1), observed.stderr
        if observed.returncode == 1 or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.1)
    assert observed.returncode == 1, f"harness processes remain: {observed.stdout}"
