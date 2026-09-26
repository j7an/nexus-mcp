import asyncio

import pytest
from claude_agent_sdk import CLIConnectionError, CLINotFoundError, ResultError, SystemMessage
from fastmcp.exceptions import ToolError

from nexus_mcp.backends import claude
from nexus_mcp.types import BackendInfo, PromptRequest, PromptResult
from tests.unit.backends.claude_fakes import assistant, factory, result


@pytest.fixture
def created():
    return []


def install(monkeypatch, created, script, exit_error=None):
    monkeypatch.setattr(claude, "ClaudeSDKClient", factory(script, created, exit_error))


def request(tmp_path, **overrides):
    return PromptRequest(prompt="do it", cwd=tmp_path, **overrides)


async def test_new_turn_returns_result(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [assistant(), result()])
    got = await claude.run(request(tmp_path))
    assert got == PromptResult(
        backend="claude", session_id="sid-1", output="done", usage={"output_tokens": 3}
    )
    [client] = created
    assert client.queries == ["do it"]
    assert client.options.resume is None and client.options.fork_session is False
    assert client.closed


async def test_continue_and_fork_set_resume_options(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [result(session_id="sid-2")])
    got = await claude.run(request(tmp_path, session_id="sid-1", fork=True, model="haiku"))
    assert got.session_id == "sid-2"
    [client] = created
    assert client.options.resume == "sid-1"
    assert client.options.fork_session is True
    assert client.options.model == "haiku"


async def test_profile_reaches_options(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [result()])
    await claude.run(request(tmp_path, profile="workspace_write"))
    assert created[0].options.sandbox is not None
    assert created[0].options.cwd == str(tmp_path)


async def test_empty_result_text_becomes_empty_output(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [result(result=None)])
    assert (await claude.run(request(tmp_path))).output == ""


@pytest.mark.parametrize(
    ("script", "match"),
    [
        ([assistant(error="authentication_failed"), result(is_error=True)], "authentication"),
        ([result(is_error=True, api_error_status=401)], "authentication"),
        ([assistant(error="rate_limit"), result(is_error=True)], "rate limited"),
        ([result(is_error=True, api_error_status=529)], "rate limited"),
        ([assistant(error="billing_error"), result(is_error=True)], "billing"),
        ([result(is_error=True, subtype="error_max_turns")], "subtype=error_max_turns"),
        ([result(is_error=True, subtype="weird provider text")], "subtype=unknown"),
    ],
)
async def test_error_results_are_classified(monkeypatch, created, tmp_path, script, match):
    install(monkeypatch, created, script)
    with pytest.raises(ToolError, match=match):
        await claude.run(request(tmp_path))


async def test_error_message_never_contains_provider_text(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [result(is_error=True, result="SECRET-provider-prose")])
    with pytest.raises(ToolError) as info:
        await claude.run(request(tmp_path))
    assert "SECRET" not in str(info.value)


async def test_result_error_is_classified(monkeypatch, created, tmp_path):
    error = ResultError(
        "boom", {"subtype": "success", "api_error_status": 429, "result": "SECRET"}, 1
    )
    install(monkeypatch, created, [result(is_error=True, api_error_status=429)], exit_error=error)
    with pytest.raises(ToolError, match="rate limited") as info:
        await claude.run(request(tmp_path))
    assert "SECRET" not in str(info.value)


async def test_cli_not_found(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [CLINotFoundError("missing")])
    with pytest.raises(ToolError, match="CLI not found"):
        await claude.run(request(tmp_path))


async def test_connection_error_is_generic(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [CLIConnectionError("SECRET detail")])
    with pytest.raises(ToolError, match=r"Claude run failed \(CLIConnectionError\)") as info:
        await claude.run(request(tmp_path))
    assert "SECRET" not in str(info.value)


async def test_missing_result_message(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [assistant()])
    with pytest.raises(ToolError, match="without a result"):
        await claude.run(request(tmp_path))


async def test_cancellation_closes_client(monkeypatch, created, tmp_path):
    started = asyncio.Event()

    async def block_forever():
        started.set()
        await asyncio.Event().wait()

    install(monkeypatch, created, [block_forever])
    task = asyncio.create_task(claude.run(request(tmp_path)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert created[0].closed


async def test_session_id_reported_once_as_soon_as_known(monkeypatch, created, tmp_path):
    init = SystemMessage(subtype="init", data={"session_id": "sid-1"})
    install(monkeypatch, created, [init, assistant(), result()])
    seen: list[str] = []
    await claude.run(request(tmp_path), seen.append)
    assert seen == ["sid-1"]


async def test_info_reports_installed():
    assert await claude.info() == BackendInfo(name="claude", installed=True, models=None)
