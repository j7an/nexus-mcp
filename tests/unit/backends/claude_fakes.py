"""Scripted stand-in for ClaudeSDKClient."""

from collections.abc import Callable
from typing import Any

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock


def assistant(value: str = "working", *, error: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=value)], model="test", error=error, session_id="sid-1"
    )


def result(**overrides: Any) -> ResultMessage:
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 1,
        "duration_api_ms": 1,
        "is_error": False,
        "num_turns": 1,
        "session_id": "sid-1",
        "result": "done",
        "usage": {"output_tokens": 3},
    }
    return ResultMessage(**(fields | overrides))


class FakeClient:
    """Yield scripted messages; exceptions in the script are raised; callables are awaited."""

    def __init__(self, options: Any, script: list[Any], exit_error: BaseException | None) -> None:
        self.options = options
        self.script = script
        self.exit_error = exit_error
        self.queries: list[str] = []
        self.closed = False

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        self.closed = True
        if self.exit_error is not None:
            raise self.exit_error
        return False

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def receive_response(self):  # type: ignore[no-untyped-def]
        for item in self.script:
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                await item()
                continue
            yield item


def factory(
    script: list[Any], created: list[FakeClient], exit_error: BaseException | None = None
) -> Callable[[Any], FakeClient]:
    def make(options: Any) -> FakeClient:
        client = FakeClient(options, script, exit_error)
        created.append(client)
        return client

    return make
