"""Antigravity SDK configuration, turn, and lifecycle contracts."""

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import anyio
import pytest
from fastmcp.exceptions import ToolError

pytest.importorskip("google.antigravity")

from google.antigravity.hooks import policy
from google.antigravity.types import (
    AntigravityConnectionError,
    AntigravityExecutionError,
    AntigravityValidationError,
    BuiltinTools,
    SessionContinuationMode,
    StopReason,
    UsageMetadata,
)

from nexus_mcp.backends import antigravity
from nexus_mcp.types import BackendInfo, PromptRequest
from tests.unit.backends.antigravity_fakes import factory

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


@pytest.fixture
def created():
    return []


def install(monkeypatch, created, **kwargs):
    monkeypatch.setattr(antigravity, "Agent", factory(created, **kwargs))


async def test_info_reports_no_model_list():
    assert await antigravity.info() == BackendInfo(name="antigravity", installed=True, models=None)


async def test_fork_is_rejected_before_agent_starts(monkeypatch, created, tmp_path):
    install(monkeypatch, created)
    message = "Antigravity does not support fork; omit fork to continue the session"
    with pytest.raises(ToolError, match=f"^{re.escape(message)}$"):
        await antigravity.run(request(tmp_path, session_id=CID, fork=True))
    assert created == []
    assert not antigravity._STORE.exists()


async def test_new_session_runs_turn(monkeypatch, created, tmp_path):
    install(monkeypatch, created, usage=UsageMetadata(total_token_count=3))
    result = await antigravity.run(request(tmp_path))
    assert result.backend == "antigravity"
    assert UUID(result.session_id).version == 4
    assert result.session_id == created[0].config.conversation_id
    assert result.output == "done"
    assert result.usage == {"total_token_count": 3}
    assert created[0].prompts == ["do it"]
    assert created[0].events == ["enter-start", "enter-done", "chat", "exit"]
    assert antigravity._STORE.is_dir()


async def test_continue_uses_given_id(monkeypatch, created, tmp_path):
    install(monkeypatch, created)
    result = await antigravity.run(request(tmp_path, session_id=CID))
    assert result.session_id == CID
    assert created[0].config.session_continuation_mode == SessionContinuationMode.RESUME


async def test_usage_none_passes_through(monkeypatch, created, tmp_path):
    install(monkeypatch, created)
    assert (await antigravity.run(request(tmp_path))).usage is None


async def test_session_reported_after_start_before_chat(monkeypatch, created, tmp_path):
    install(monkeypatch, created)
    seen = []

    def on_session(session_id):
        seen.append((session_id, created[0].events[:]))

    result = await antigravity.run(request(tmp_path), on_session)
    assert seen == [(result.session_id, ["enter-start", "enter-done"])]


@pytest.mark.parametrize("sandbox", [None, SimpleNamespace(available=False)])
async def test_workspace_write_requires_sandbox(monkeypatch, created, tmp_path, sandbox):
    install(monkeypatch, created, sandbox=sandbox)
    message = (
        "Antigravity OS sandbox unavailable; workspace_write cannot confine commands. "
        "Use read_only, or full_access to run unsandboxed"
    )
    with pytest.raises(ToolError, match=f"^{re.escape(message)}$") as raised:
        await antigravity.run(request(tmp_path, profile="workspace_write"))
    assert "SECRET-provider" not in str(raised.value)
    assert created[0].events == ["enter-start", "enter-done", "exit"]


async def test_workspace_write_with_available_sandbox_runs(monkeypatch, created, tmp_path):
    install(monkeypatch, created)
    assert (await antigravity.run(request(tmp_path, profile="workspace_write"))).output == "done"


@pytest.mark.parametrize("profile", ["read_only", "full_access"])
async def test_unsandboxed_profiles_run(monkeypatch, created, tmp_path, profile):
    install(monkeypatch, created, sandbox=None)
    assert (await antigravity.run(request(tmp_path, profile=profile))).output == "done"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            AntigravityValidationError("SECRET-provider"),
            (
                "Antigravity needs credentials: set GEMINI_API_KEY, or configure Vertex "
                "(GOOGLE_GENAI_USE_VERTEXAI with project/location or an API key)"
            ),
        ),
        (
            RuntimeError("SECRET-provider"),
            (
                "Antigravity could not start or resume the session "
                "(unknown session_id, or the harness failed to start)"
            ),
        ),
    ],
)
async def test_startup_errors_are_classified(monkeypatch, created, tmp_path, error, message):
    install(monkeypatch, created, start_error=error)
    seen = []
    with pytest.raises(ToolError, match=f"^{re.escape(message)}$") as raised:
        await antigravity.run(request(tmp_path), seen.append)
    assert "SECRET-provider" not in str(raised.value)
    assert created[0].events == ["enter-start", "exit"]
    assert seen == []


@pytest.mark.parametrize("error_type", [AntigravityConnectionError, AntigravityExecutionError])
async def test_turn_errors_are_classified(monkeypatch, created, tmp_path, error_type):
    install(monkeypatch, created, chat_error=error_type("SECRET-provider"))
    message = f"Antigravity run failed ({error_type.__name__}); check credentials and model"
    with pytest.raises(ToolError, match=f"^{re.escape(message)}$") as raised:
        await antigravity.run(request(tmp_path))
    assert "SECRET-provider" not in str(raised.value)
    assert created[0].events == ["enter-start", "enter-done", "chat", "exit"]


async def test_turn_runtime_error_is_not_classified_as_startup(monkeypatch, created, tmp_path):
    install(monkeypatch, created, chat_error=RuntimeError("SECRET-provider"))
    with pytest.raises(RuntimeError, match="^SECRET-provider$"):
        await antigravity.run(request(tmp_path))
    assert "exit" in created[0].events


async def test_quota_exhausted_without_exception_is_tool_error(monkeypatch, created, tmp_path):
    install(monkeypatch, created, stop_reason=StopReason.QUOTA_EXHAUSTED)
    with pytest.raises(ToolError, match="^Antigravity quota exhausted; retry later$") as raised:
        await antigravity.run(request(tmp_path))
    assert "SECRET-provider" not in str(raised.value)
    assert "exit" in created[0].events


async def test_blank_prompt_is_tool_error(monkeypatch, created, tmp_path):
    install(monkeypatch, created, chat_error=ValueError("chat() requires ... SECRET-provider"))
    message = "Antigravity rejected the prompt (empty after trimming whitespace)"
    with pytest.raises(ToolError, match=f"^{re.escape(message)}$") as raised:
        await antigravity.run(request(tmp_path))
    assert "SECRET-provider" not in str(raised.value)
    assert "exit" in created[0].events


async def test_conversation_id_drift_is_tool_error(monkeypatch, created, tmp_path):
    install(monkeypatch, created, drift_id="SECRET-provider")
    message = (
        "Antigravity returned an unexpected conversation id; "
        "refusing to report an unresumable session"
    )
    with pytest.raises(ToolError, match=f"^{re.escape(message)}$") as raised:
        await antigravity.run(request(tmp_path))
    assert "SECRET-provider" not in str(raised.value)
    assert "exit" in created[0].events


async def test_unwritable_store_is_tool_error(monkeypatch, created, tmp_path):
    install(monkeypatch, created)
    (tmp_path / "file").write_text("occupied")
    monkeypatch.setattr(antigravity, "_STORE", tmp_path / "file" / "store")
    message = "Antigravity session store ~/.nexus-mcp/antigravity is not writable"
    with pytest.raises(ToolError, match=f"^{re.escape(message)}$") as raised:
        await antigravity.run(request(tmp_path))
    assert "SECRET-provider" not in str(raised.value)
    assert created == []


async def test_cancel_during_chat_exits_agent(monkeypatch, created, tmp_path):
    started = asyncio.Event()
    install(monkeypatch, created, block_chat=started)
    task = asyncio.create_task(antigravity.run(request(tmp_path)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert created[0].events == ["enter-start", "enter-done", "chat", "exit"]


async def test_anyio_cancel_during_chat_exits_agent(monkeypatch, created, tmp_path):
    started = asyncio.Event()
    install(monkeypatch, created, block_chat=started)

    async def cancel_mid_turn(scope):
        await started.wait()
        scope.cancel()

    with anyio.CancelScope() as scope:
        canceller = asyncio.ensure_future(cancel_mid_turn(scope))
        await antigravity.run(request(tmp_path))
    await canceller
    assert scope.cancelled_caught
    assert created[0].events == ["enter-start", "enter-done", "chat", "exit"]


async def test_cancel_during_startup_exits_after_start(monkeypatch, created, tmp_path):
    gate = asyncio.Event()
    install(monkeypatch, created, enter_gate=gate)
    task = asyncio.create_task(antigravity.run(request(tmp_path)))
    while not created:
        if task.done():
            await task
        await asyncio.sleep(0)
    await created[0].entering.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert created[0].events == ["enter-start"]
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert created[0].events == ["enter-start", "enter-done", "exit"]


async def test_anyio_cancel_during_startup_exits_after_start(monkeypatch, created, tmp_path):
    gate = asyncio.Event()
    install(monkeypatch, created, enter_gate=gate)

    async def cancel_while_starting(scope):
        while not created:
            await asyncio.sleep(0)
        await created[0].entering.wait()
        scope.cancel()
        await asyncio.sleep(0.05)
        gate.set()

    with anyio.CancelScope() as scope:
        canceller = asyncio.ensure_future(cancel_while_starting(scope))
        await antigravity.run(request(tmp_path))
    await canceller
    assert scope.cancelled_caught
    assert created[0].events == ["enter-start", "enter-done", "exit"]
