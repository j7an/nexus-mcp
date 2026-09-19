"""Scripted stand-ins for ClaudeSDKClient and the worker execution context."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from nexus_mcp.core import (
    BackendEvent,
    InputRequest,
    InputResponse,
    JobAttempt,
    ProviderReference,
    ResolvedExecutionConfig,
)
from tests.fixtures import make_agent_job, make_agent_session, make_workspace


def assistant(*blocks: Any, error: str | None = None) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="test", error=error, session_id="sid-1")


def text(value: str) -> AssistantMessage:
    return assistant(TextBlock(text=value))


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
    """Yield scripted SDK messages; scripted callables run mid-stream with the client."""

    def __init__(self, options: Any, script: list[Any]) -> None:
        self.options = options
        self.script = script
        self.queries: list[str] = []
        self.interrupted = False
        self.closed = False
        self.drained = False

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        self.closed = True
        return False

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def interrupt(self) -> None:
        self.interrupted = True

    async def receive_response(self):  # type: ignore[no-untyped-def]
        for item in self.script:
            if isinstance(item, BaseException):
                raise item
            if callable(item):
                produced = await item(self)
                if produced is not None:
                    yield produced
                continue
            yield item
        self.drained = True


def factory(script: list[Any], created: list[FakeClient]) -> Callable[[Any], FakeClient]:
    def make(options: Any) -> FakeClient:
        client = FakeClient(options, script)
        created.append(client)
        return client

    return make


class FakeContext:
    """Record backend effects; control signals and input answers are scripted."""

    def __init__(
        self,
        *,
        workspace_path: Path,
        resolved_config: ResolvedExecutionConfig | None = None,
        **job_overrides: Any,
    ) -> None:
        self.job = make_agent_job(backend_id="claude", **job_overrides)
        self.attempt = JobAttempt(job_id=self.job.job_id, attempt_number=1)
        self.workspace = make_workspace(canonical_path=workspace_path)
        self.resolved_config = resolved_config or ResolvedExecutionConfig()
        self.session = make_agent_session(
            backend_id="claude", session_id="session-test", parent_session_id=None
        )
        self.events: list[BackendEvent] = []
        self.deltas: list[str] = []
        self.references: list[ProviderReference] = []
        self.input_requests: list[InputRequest] = []
        self.input_response: InputResponse | None = None
        self.control: asyncio.Queue[Any] = asyncio.Queue()

    async def emit(self, event: BackendEvent) -> None:
        self.events.append(event)

    async def emit_output_delta(self, text_value: str) -> None:
        self.deltas.append(text_value)

    async def record_provider_reference(self, reference: ProviderReference) -> None:
        self.references.append(reference)

    async def request_input(self, request: InputRequest) -> InputResponse:
        self.input_requests.append(request)
        assert self.input_response is not None
        return self.input_response

    async def wait_for_control(self) -> Any:
        return await self.control.get()

    async def checkpoint(self) -> None:
        return
