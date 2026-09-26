"""Scripted stand-ins for openai_codex.AsyncCodex, threads, and turn handles."""

import asyncio
from typing import Any

import openai_codex.generated.v2_all as wire
from openai_codex.models import Notification

TURN = "turn-1"


def agent_message(text: str, phase: str | None = "final_answer") -> Notification:
    item = wire.AgentMessageThreadItem(
        id=f"item-{text}",
        text=text,
        phase=None if phase is None else wire.MessagePhase(phase),
        type="agentMessage",
    )
    payload = wire.ItemCompletedNotification(
        completed_at_ms=0, thread_id="t", turn_id=TURN, item=wire.ThreadItem(item)
    )
    return Notification(method="item/completed", payload=payload)


def usage(total: int = 3) -> Notification:
    breakdown = wire.TokenUsageBreakdown(
        cached_input_tokens=0,
        input_tokens=1,
        output_tokens=total - 1,
        reasoning_output_tokens=0,
        total_tokens=total,
    )
    payload = wire.ThreadTokenUsageUpdatedNotification(
        thread_id="t",
        turn_id=TURN,
        token_usage=wire.ThreadTokenUsage(last=breakdown, total=breakdown),
    )
    return Notification(method="thread/tokenUsage/updated", payload=payload)


def completed(status: str = "completed", info: Any = None, message: str = "x") -> Notification:
    error = None
    if status == "failed":
        error = wire.TurnError(
            message=message, codex_error_info=None if info is None else wire.CodexErrorInfo(info)
        )
    turn = wire.Turn(id=TURN, items=[], status=wire.TurnStatus(status), error=error)
    return Notification(
        method="turn/completed", payload=wire.TurnCompletedNotification(thread_id="t", turn=turn)
    )


class FakeHandle:
    def __init__(self, script: list[Any], interrupt_error: BaseException | None) -> None:
        self.script = script
        self.interrupt_error = interrupt_error
        self.interrupted = False
        self.interrupt_completed = False

    async def stream(self):  # type: ignore[no-untyped-def]
        for item in self.script:
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                await item()
                continue
            yield item

    async def interrupt(self) -> None:
        self.interrupted = True
        await asyncio.sleep(0)
        if self.interrupt_error is not None:
            raise self.interrupt_error
        self.interrupt_completed = True


class FakeThread:
    def __init__(self, thread_id: str, handle: FakeHandle) -> None:
        self.id = thread_id
        self.handle = handle
        self.turns: list[tuple[str, dict[str, Any]]] = []

    async def turn(self, prompt: str, **kwargs: Any) -> FakeHandle:
        self.turns.append((prompt, kwargs))
        return self.handle


class FakeCodex:
    """Records thread calls; `enter_error` simulates an app-server launch failure and
    `enter_gate` holds startup open so tests can cancel during it."""

    def __init__(
        self,
        config: Any,
        script: list[Any],
        enter_error: BaseException | None,
        interrupt_error: BaseException | None,
        models: list[str],
        enter_gate: asyncio.Event | None,
    ) -> None:
        self.config = config
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.handle = FakeHandle(script, interrupt_error)
        self.thread: FakeThread | None = None
        self.enter_error = enter_error
        self.enter_gate = enter_gate
        self.entering = asyncio.Event()
        self.model_ids = models
        self.events: list[str] = []
        self.closed = False

    async def __aenter__(self) -> "FakeCodex":
        self.events.append("enter-start")
        self.entering.set()
        if self.enter_gate is not None:
            await self.enter_gate.wait()
        if self.enter_error is not None:
            raise self.enter_error
        self.events.append("enter-done")
        return self

    async def close(self) -> None:
        self.events.append("close")
        self.closed = True

    def _thread(
        self, name: str, thread_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> FakeThread:
        self.calls.append((name, args, kwargs))
        self.thread = FakeThread(thread_id, self.handle)
        return self.thread

    async def thread_start(self, **kwargs: Any) -> FakeThread:
        return self._thread("start", "thread-new", (), kwargs)

    async def thread_resume(self, thread_id: str, **kwargs: Any) -> FakeThread:
        return self._thread("resume", thread_id, (thread_id,), kwargs)

    async def thread_fork(self, thread_id: str, **kwargs: Any) -> FakeThread:
        return self._thread("fork", "thread-forked", (thread_id,), kwargs)

    async def models(self, *, include_hidden: bool = False) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(data=[SimpleNamespace(id=model) for model in self.model_ids])


def factory(
    script: list[Any],
    created: list[FakeCodex],
    *,
    enter_error: BaseException | None = None,
    interrupt_error: BaseException | None = None,
    models: list[str] | None = None,
    enter_gate: asyncio.Event | None = None,
):  # type: ignore[no-untyped-def]
    def make(config: Any) -> FakeCodex:
        codex = FakeCodex(config, script, enter_error, interrupt_error, models or [], enter_gate)
        created.append(codex)
        return codex

    return make


async def block_forever(started: asyncio.Event) -> None:
    started.set()
    await asyncio.Event().wait()
