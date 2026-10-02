"""SDK-free doubles for Antigravity turn and lifecycle tests."""

import asyncio
from types import SimpleNamespace
from typing import Any

_AVAILABLE = SimpleNamespace(available=True)


class FakeResponse:
    def __init__(self, text: str, usage: Any, stop_reason: Any) -> None:
        self._text = text
        self.usage_metadata = usage
        self.stop_reason = stop_reason

    async def text(self) -> str:
        return self._text


class FakeAgent:
    def __init__(
        self,
        config: Any,
        *,
        sandbox: Any,
        text: str,
        usage: Any,
        start_error: Exception | None,
        chat_error: Exception | None,
        drift_id: str | None,
        enter_gate: asyncio.Event | None,
        block_chat: asyncio.Event | None,
        stop_reason: Any,
    ) -> None:
        self.config = config
        self.events: list[str] = []
        self.entering = asyncio.Event()
        self.sandbox_status = sandbox
        self.conversation_id: str | None = None
        self.start_error = start_error
        self.chat_error = chat_error
        self.drift_id = drift_id
        self.enter_gate = enter_gate
        self.block_chat = block_chat
        self.response = FakeResponse(text, usage, stop_reason)
        self.prompts: list[str] = []

    async def __aenter__(self) -> "FakeAgent":
        self.events.append("enter-start")
        self.entering.set()
        if self.enter_gate is not None:
            await self.enter_gate.wait()
        if self.start_error is not None:
            raise self.start_error
        self.events.append("enter-done")
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await asyncio.sleep(0)  # Detect cleanup interrupted by an AnyIO cancel scope.
        self.events.append("exit")

    async def chat(self, prompt: str) -> FakeResponse:
        self.events.append("chat")
        self.prompts.append(prompt)
        if self.chat_error is not None:
            raise self.chat_error
        if self.block_chat is not None:
            self.block_chat.set()
            await asyncio.Event().wait()
        self.conversation_id = self.drift_id or self.config.conversation_id
        return self.response


def factory(
    created: list[FakeAgent],
    *,
    sandbox: Any = _AVAILABLE,
    text: str = "done",
    usage: Any = None,
    start_error: Exception | None = None,
    chat_error: Exception | None = None,
    drift_id: str | None = None,
    enter_gate: asyncio.Event | None = None,
    block_chat: asyncio.Event | None = None,
    stop_reason: Any = "UNSPECIFIED",
):  # type: ignore[no-untyped-def]
    def make(config: Any) -> FakeAgent:
        agent = FakeAgent(
            config,
            sandbox=sandbox,
            text=text,
            usage=usage,
            start_error=start_error,
            chat_error=chat_error,
            drift_id=drift_id,
            enter_gate=enter_gate,
            block_chat=block_chat,
            stop_reason=stop_reason,
        )
        created.append(agent)
        return agent

    return make
