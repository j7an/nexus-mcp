import asyncio
import functools
import io
import threading

import anyio
import openai_codex.generated.v2_all as wire
import pytest
from fastmcp.exceptions import ToolError
from openai_codex import ApprovalMode, CodexError, Sandbox, TransportClosedError
from openai_codex.models import InitializeResponse

from nexus_mcp.backends import codex
from nexus_mcp.types import BackendInfo, PromptRequest
from tests.unit.backends.codex_fakes import (
    agent_message,
    block_forever,
    completed,
    factory,
    usage,
)


@pytest.fixture
def created():
    return []


def install(monkeypatch, created, script, **kwargs):
    monkeypatch.setattr(codex, "AsyncCodex", factory(script, created, **kwargs))
    monkeypatch.setattr(codex, "_close_sdk_pipes", lambda _client: None)


def request(tmp_path, **overrides):
    return PromptRequest(prompt="do it", cwd=tmp_path, **overrides)


async def test_new_thread_turn(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [agent_message("done"), usage(3), completed()])
    got = await codex.run(request(tmp_path, model="gpt-5.4"))
    assert got.backend == "codex" and got.session_id == "thread-new" and got.output == "done"
    assert got.usage is not None and got.usage["total"]["total_tokens"] == 3
    [client] = created
    assert client.config.experimental_api is False
    [(name, args, kwargs)] = client.calls
    assert name == "start" and args == ()
    assert kwargs == {
        "cwd": str(tmp_path),
        "model": "gpt-5.4",
        "sandbox": Sandbox.read_only,
        "approval_mode": ApprovalMode.deny_all,
    }
    assert client.thread.turns[0][0] == "do it"
    assert client.closed


@pytest.mark.parametrize(
    ("fork", "expected_call", "expected_id"),
    [(False, "resume", "thread-9"), (True, "fork", "thread-forked")],
)
async def test_continue_and_fork(monkeypatch, created, tmp_path, fork, expected_call, expected_id):
    install(monkeypatch, created, [agent_message("ok"), completed()])
    got = await codex.run(request(tmp_path, session_id="thread-9", fork=fork))
    [(name, args, _kwargs)] = created[0].calls
    assert (name, args) == (expected_call, ("thread-9",))
    assert got.session_id == expected_id


@pytest.mark.parametrize(
    ("profile", "sandbox"),
    [
        ("read_only", Sandbox.read_only),
        ("workspace_write", Sandbox.workspace_write),
        ("full_access", Sandbox.full_access),
    ],
)
async def test_profile_sent_at_thread_and_turn_level(
    monkeypatch, created, tmp_path, profile, sandbox
):
    install(monkeypatch, created, [agent_message("ok"), completed()])
    await codex.run(request(tmp_path, profile=profile, session_id="thread-9"))
    client = created[0]
    assert client.calls[0][2]["sandbox"] is sandbox
    turn_kwargs = client.thread.turns[0][1]
    assert turn_kwargs["sandbox"] is sandbox
    assert turn_kwargs["approval_mode"] is ApprovalMode.deny_all
    assert turn_kwargs["cwd"] == str(tmp_path)


async def test_final_answer_selection(monkeypatch, created, tmp_path):
    install(
        monkeypatch,
        created,
        [
            agent_message("thinking", phase="commentary"),
            agent_message("unphased", phase=None),
            agent_message("final", phase="final_answer"),
            agent_message("late commentary", phase="commentary"),
            completed(),
        ],
    )
    assert (await codex.run(request(tmp_path))).output == "final"


async def test_unphased_message_is_fallback(monkeypatch, created, tmp_path):
    install(
        monkeypatch,
        created,
        [
            agent_message("first", phase=None),
            agent_message("second", phase=None),
            agent_message("note", phase="commentary"),
            completed(),
        ],
    )
    assert (await codex.run(request(tmp_path))).output == "second"


async def test_no_agent_message_is_error(monkeypatch, created, tmp_path):
    install(
        monkeypatch, created, [agent_message("only commentary", phase="commentary"), completed()]
    )
    with pytest.raises(ToolError, match="without a final answer"):
        await codex.run(request(tmp_path))


async def test_missing_completion_is_error(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [agent_message("done")])
    with pytest.raises(ToolError, match="without completing"):
        await codex.run(request(tmp_path))


@pytest.mark.parametrize(
    ("info", "match"),
    [
        (wire.CodexErrorInfoValue.unauthorized, "authentication"),
        (wire.CodexErrorInfoValue.usage_limit_exceeded, "limit"),
        (wire.CodexErrorInfoValue.rate_limit_exceeded, "limit"),
        (wire.CodexErrorInfoValue.server_overloaded, "limit"),
        (wire.CodexErrorInfoValue.context_window_exceeded, r"kind=contextWindowExceeded"),
        (
            wire.HttpConnectionFailedCodexErrorInfo.model_validate(
                {"httpConnectionFailed": {"httpStatusCode": 502}}
            ),
            r"kind=HttpConnectionFailedCodexErrorInfo",
        ),
        (None, r"Codex turn failed$"),
    ],
)
async def test_failed_turn_errors_are_classified_without_provider_text(
    monkeypatch, created, tmp_path, info, match
):
    install(monkeypatch, created, [completed("failed", info=info, message="SECRET-provider")])
    with pytest.raises(ToolError, match=match) as raised:
        await codex.run(request(tmp_path))
    assert "SECRET" not in str(raised.value)


async def test_interrupted_turn(monkeypatch, created, tmp_path):
    install(monkeypatch, created, [completed("interrupted")])
    with pytest.raises(ToolError, match="interrupted"):
        await codex.run(request(tmp_path))


async def test_cancel_interrupts_turn(monkeypatch, created, tmp_path):
    started = asyncio.Event()
    install(monkeypatch, created, [functools.partial(block_forever, started)])
    task = asyncio.create_task(codex.run(request(tmp_path)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert created[0].handle.interrupted
    assert created[0].closed


async def test_cancel_survives_interrupt_failure(monkeypatch, created, tmp_path):
    started = asyncio.Event()
    install(
        monkeypatch,
        created,
        [functools.partial(block_forever, started)],
        interrupt_error=TransportClosedError("gone"),
    )
    task = asyncio.create_task(codex.run(request(tmp_path)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_anyio_cancel_waits_for_interrupt(monkeypatch, created, tmp_path):
    started = asyncio.Event()
    install(monkeypatch, created, [functools.partial(block_forever, started)])

    async def cancel_mid_turn(scope: anyio.CancelScope) -> None:
        await started.wait()
        scope.cancel()

    with anyio.CancelScope() as scope:
        canceller = asyncio.ensure_future(cancel_mid_turn(scope))
        await codex.run(request(tmp_path))
    await canceller
    assert scope.cancelled_caught
    assert created[0].handle.interrupt_completed
    assert created[0].closed


async def test_cancel_during_startup_closes_after_launch(monkeypatch, created, tmp_path):
    gate = asyncio.Event()
    install(monkeypatch, created, [], enter_gate=gate)
    task = asyncio.create_task(codex.run(request(tmp_path)))
    while not created:
        await asyncio.sleep(0)
    await created[0].entering.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert created[0].events == ["enter-start"]  # close must wait for the launch to finish
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert created[0].events == ["enter-start", "enter-done", "close"]


async def test_anyio_cancel_during_startup_closes_after_launch(monkeypatch, created, tmp_path):
    """MCP cancels requests with an AnyIO cancel scope, which re-cancels every await."""
    gate = asyncio.Event()
    install(monkeypatch, created, [], enter_gate=gate)

    async def cancel_while_starting(scope: anyio.CancelScope) -> None:
        while not created:
            await asyncio.sleep(0)
        await created[0].entering.wait()
        scope.cancel()
        await asyncio.sleep(0.05)  # let the cancelled run() reach its cleanup first
        gate.set()

    with anyio.CancelScope() as scope:
        canceller = asyncio.ensure_future(cancel_while_starting(scope))
        await codex.run(request(tmp_path))
    await canceller
    assert scope.cancelled_caught
    assert created[0].events == ["enter-start", "enter-done", "close"]


async def test_session_reported_before_turn_output(monkeypatch, created, tmp_path):
    started = asyncio.Event()
    install(monkeypatch, created, [functools.partial(block_forever, started)])
    seen: list[str] = []
    task = asyncio.create_task(codex.run(request(tmp_path), seen.append))
    await started.wait()
    assert seen == ["thread-new"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (FileNotFoundError("SECRET/path/codex"), "binary not found"),
        (TransportClosedError("SECRET"), r"Codex run failed \(TransportClosedError\)"),
        (CodexError("SECRET"), r"Codex run failed \(CodexError\)"),
    ],
)
async def test_launch_failure(monkeypatch, created, tmp_path, error, match):
    install(monkeypatch, created, [], enter_error=error)
    with pytest.raises(ToolError, match=match) as raised:
        await codex.run(request(tmp_path))
    assert "SECRET" not in str(raised.value)


async def test_info_lists_models(monkeypatch, created):
    install(monkeypatch, created, [], models=["gpt-5.4", "gpt-5.4-mini"])
    assert await codex.info() == BackendInfo(
        name="codex", installed=True, models=["gpt-5.4", "gpt-5.4-mini"]
    )


async def test_info_survives_launch_failure(monkeypatch, created):
    install(monkeypatch, created, [], enter_error=FileNotFoundError("x"))
    got = await codex.info()
    assert got.installed is True and got.models is None and "could not start" in got.hint.lower()


@pytest.mark.parametrize("exit_kind", ["normal", "startup_error", "cancel_startup"])
async def test_sdk_pipes_close_after_process_exit(monkeypatch, exit_kind):
    """Exercise the real SDK close path with a process that records shutdown order."""
    events: list[str] = []

    class Pipe(io.StringIO):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def close(self) -> None:
            events.append(f"close-{self.name}")
            super().close()

    class Process:
        def __init__(self) -> None:
            self.stdin = Pipe("stdin")
            self.stdout = Pipe("stdout")
            self.stderr = Pipe("stderr")
            self.exited = False

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout=None) -> int:
            events.append("wait")
            self.exited = True
            return 0

        def poll(self) -> int | None:
            return 0 if self.exited else None

    process = Process()
    sdk_type = codex.AsyncCodex
    client = sdk_type()
    sync = client._client._sync
    sync._proc = process
    entered = threading.Event()
    release = threading.Event()

    def initialize() -> InitializeResponse:
        entered.set()
        if exit_kind == "cancel_startup":
            release.wait(timeout=2)
        if exit_kind == "startup_error":
            raise RuntimeError("initialize failed")
        return InitializeResponse(userAgent="fake/1")

    monkeypatch.setattr(sync, "initialize", initialize)
    monkeypatch.setattr(codex, "AsyncCodex", lambda _config: client)

    async def open_session() -> None:
        async with codex._session():
            pass

    if exit_kind == "cancel_startup":
        task = asyncio.create_task(open_session())
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif exit_kind == "startup_error":
        with pytest.raises(RuntimeError, match="initialize failed"):
            await open_session()
    else:
        await open_session()

    assert process.exited
    assert process.stdout.closed and process.stderr.closed
    assert events.index("wait") < events.index("close-stdout")
    assert events.index("wait") < events.index("close-stderr")
