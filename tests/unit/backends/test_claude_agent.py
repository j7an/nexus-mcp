"""Claude Agent backend turn and configuration behavior."""

import asyncio

import pytest
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext

from nexus_mcp.backends.base import BackendFailure, CancelRequested, InputResolved
from nexus_mcp.backends.claude_agent import ClaudeAgentBackend
from nexus_mcp.core import (
    PermissionRequest,
    PermissionResponse,
    ProviderReference,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    TurnOperation,
    TurnResult,
)
from tests.fixtures import make_workspace
from tests.unit.backends.claude_fakes import FakeClient, FakeContext, factory, result, text


def test_descriptor_matches_spec():
    descriptor = ClaudeAgentBackend().descriptor
    capabilities = descriptor.capabilities
    assert descriptor.backend_id == "claude"
    assert descriptor.display_name == "Claude Agent"
    assert "Claude Code" not in (descriptor.description or "")
    assert capabilities.operations == frozenset({"turn", "fork", "review"})
    assert capabilities.cancellation and capabilities.graceful_interrupt
    assert capabilities.session_continuation and capabilities.session_fork
    assert capabilities.input_required and capabilities.structured_output
    assert capabilities.dynamic_models is False
    assert capabilities.sandbox_modes == frozenset(
        {"read_only", "workspace_write", "danger_full_access"}
    )
    assert capabilities.review_targets == frozenset({"working_tree", "branch", "commit"})
    assert capabilities.review_deliveries == frozenset({"inline", "detached"})


async def test_config_defaults_to_read_only_on_request(tmp_path):
    resolved = await ClaudeAgentBackend().resolve_execution_config(
        RequestedExecutionConfig(), make_workspace(canonical_path=tmp_path)
    )
    assert resolved.sandbox == "read_only"
    assert resolved.approval_policy == "on_request"
    assert resolved.timeout_seconds is not None


async def test_availability_makes_no_auth_claim(tmp_path):
    availability = await ClaudeAgentBackend().check_availability(
        make_workspace(canonical_path=tmp_path)
    )
    assert availability.available is True
    assert availability.authenticated is None
    assert availability.version is not None and availability.version.startswith("sdk ")


async def test_availability_reports_missing_effective_cli_without_sdk_error(tmp_path, monkeypatch):
    from claude_agent_sdk import CLINotFoundError
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    def missing_cli(_transport):
        raise CLINotFoundError("SDK-secret-cli-path")

    monkeypatch.setattr("nexus_mcp.backends.claude_agent.get_agent_env", lambda *_: None)
    monkeypatch.setattr(SubprocessCLITransport, "_find_cli", missing_cli)

    availability = await ClaudeAgentBackend().check_availability(
        make_workspace(canonical_path=tmp_path)
    )

    assert availability.available is False
    assert availability.authenticated is None
    assert availability.reason is not None
    assert availability.setup_guidance is not None
    assert "SDK-secret" not in availability.model_dump_json()


async def test_new_turn_streams_records_session_and_returns_result(tmp_path):
    created: list[FakeClient] = []
    backend = ClaudeAgentBackend(factory([text("hel"), text("lo"), result()], created))
    context = FakeContext(workspace_path=tmp_path)

    outcome = await backend.execute(TurnOperation(prompt="hi", file_refs=("a.py",)), context)

    assert isinstance(outcome, TurnResult)
    assert outcome.message == "done"
    assert outcome.usage == {"output_tokens": 3}
    assert context.deltas == ["hel", "lo"]
    assert context.references == [ProviderReference(kind="session", value="sid-1")]
    client = created[0]
    assert client.queries == ["hi\n\nFile references:\n- a.py"]
    assert client.options.resume is None
    assert client.options.cwd == str(tmp_path)
    assert client.closed


async def test_turn_records_only_first_distinct_session_id(tmp_path):
    context = FakeContext(workspace_path=tmp_path)
    backend = ClaudeAgentBackend(factory([text("first"), result(session_id="sid-2")], []))

    await backend.execute(TurnOperation(prompt="hi"), context)

    assert context.references == [ProviderReference(kind="session", value="sid-1")]


async def test_continue_resumes_checkpoint_session(tmp_path):
    created: list[FakeClient] = []
    backend = ClaudeAgentBackend(factory([result()], created))
    context = FakeContext(
        workspace_path=tmp_path,
        source_checkpoint=(ProviderReference(kind="session", value="sid-1"),),
    )
    await backend.execute(TurnOperation(prompt="again"), context)
    assert created[0].options.resume == "sid-1"


async def test_structured_turn_passes_schema_and_returns_output(tmp_path):
    created: list[FakeClient] = []
    backend = ClaudeAgentBackend(factory([result(structured_output={"n": 1})], created))
    schema = {"type": "object"}
    outcome = await backend.execute(
        TurnOperation(prompt="x", output_schema=schema), FakeContext(workspace_path=tmp_path)
    )
    assert created[0].options.output_format == {"type": "json_schema", "schema": schema}
    assert outcome.structured_output == {"n": 1}


async def test_structured_turn_without_output_is_invalid(tmp_path):
    backend = ClaudeAgentBackend(factory([result(structured_output=None)], []))
    with pytest.raises(BackendFailure) as raised:
        await backend.execute(
            TurnOperation(prompt="x", output_schema={"type": "object"}),
            FakeContext(workspace_path=tmp_path),
        )
    assert raised.value.error.code == "structured_output_invalid"
    assert raised.value.retry_disposition == "terminal"


SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("returned", [{"n": "SECRET-not-an-integer"}, {}, {"n": 1, "x": 2}, [1]])
async def test_structured_output_violating_the_schema_is_invalid(tmp_path, returned):
    """The SDK types structured_output as Any; a present value is not a conforming value."""
    with pytest.raises(BackendFailure) as raised:
        await ClaudeAgentBackend(factory([result(structured_output=returned)], [])).execute(
            TurnOperation(prompt="x", output_schema=SCHEMA), FakeContext(workspace_path=tmp_path)
        )
    assert raised.value.error.code == "structured_output_invalid"
    assert raised.value.retry_disposition == "terminal"
    assert "SECRET" not in raised.value.error.model_dump_json()
    assert raised.value.__cause__ is None


async def test_structured_output_conforming_to_the_schema_is_returned(tmp_path):
    outcome = await ClaudeAgentBackend(factory([result(structured_output={"n": 1})], [])).execute(
        TurnOperation(prompt="x", output_schema=SCHEMA), FakeContext(workspace_path=tmp_path)
    )
    assert outcome.structured_output == {"n": 1}


async def test_message_is_truncated_to_output_limit(tmp_path):
    backend = ClaudeAgentBackend(factory([result(result="é" * 10)], []))
    context = FakeContext(
        workspace_path=tmp_path, resolved_config=ResolvedExecutionConfig(output_limit_bytes=5)
    )
    outcome = await backend.execute(TurnOperation(prompt="x"), context)
    assert outcome.message == "éé"


def _hook_input(tool, tool_input):
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}


async def test_pre_tool_use_hook_gates_calls_an_allow_rule_would_approve(tmp_path, monkeypatch):
    """Settings permissions.allow rules skip can_use_tool; the hook must still decide."""
    monkeypatch.setattr("sys.platform", "linux")
    decisions: list = []

    async def call_hook(client):
        [matcher] = client.options.hooks["PreToolUse"]
        for tool, tool_input in [
            ("WebFetch", {"url": "https://x"}),
            ("Write", {"file_path": "inside.txt"}),
        ]:
            decisions.append(await matcher.hooks[0](_hook_input(tool, tool_input), "t1", None))

    context = FakeContext(
        workspace_path=tmp_path,
        resolved_config=ResolvedExecutionConfig(
            sandbox="workspace_write", approval_policy="on_request"
        ),
    )
    await ClaudeAgentBackend(factory([call_hook, result()], [])).execute(
        TurnOperation(prompt="x"), context
    )
    assert decisions[0]["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert decisions[1] == {}


async def test_pre_tool_use_hook_denies_writes_when_read_only(tmp_path):
    decisions: list = []

    async def call_hook(client):
        [matcher] = client.options.hooks["PreToolUse"]
        decisions.append(
            await matcher.hooks[0](_hook_input("Write", {"file_path": "a.txt"}), "t1", None)
        )

    await ClaudeAgentBackend(factory([call_hook, result()], [])).execute(
        TurnOperation(prompt="x"), FakeContext(workspace_path=tmp_path)
    )
    assert decisions[0]["hookSpecificOutput"]["permissionDecision"] == "deny"


async def test_changed_files_only_after_successful_tool_result(tmp_path):
    from claude_agent_sdk import ToolResultBlock, ToolUseBlock, UserMessage

    from tests.unit.backends.claude_fakes import assistant

    script = [
        assistant(
            ToolUseBlock(id="ok", name="Write", input={"file_path": "kept.txt"}),
            ToolUseBlock(id="bad", name="Write", input={"file_path": "failed.txt"}),
            ToolUseBlock(id="denied", name="Edit", input={"file_path": "denied.txt"}),
        ),
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="ok", content="done"),
                ToolResultBlock(tool_use_id="bad", content="boom", is_error=True),
                ToolResultBlock(tool_use_id="denied", content="denied", is_error=True),
            ]
        ),
        result(),
    ]
    context = FakeContext(workspace_path=tmp_path)
    outcome = await ClaudeAgentBackend(factory(script, [])).execute(
        TurnOperation(prompt="x"), context
    )
    assert outcome.changed_files == ("kept.txt",)
    assert [e.payload["path"] for e in context.events if e.type == "file_change"] == ["kept.txt"]


async def test_command_event_never_carries_the_command_line(tmp_path):
    from claude_agent_sdk import ToolUseBlock

    from tests.unit.backends.claude_fakes import assistant

    sentinel = "sk-SENTINEL-DO-NOT-LEAK"
    script = [
        assistant(
            ToolUseBlock(
                id="c1",
                name="Bash",
                input={"command": f"TOKEN={sentinel} curl -H 'Authorization: {sentinel}' x"},
            )
        ),
        result(),
    ]
    context = FakeContext(workspace_path=tmp_path)
    await ClaudeAgentBackend(factory(script, [])).execute(TurnOperation(prompt="x"), context)
    [event] = [e for e in context.events if e.type == "command"]
    assert event.payload == {"command": "curl"}
    assert sentinel not in event.model_dump_json()


async def test_cancel_interrupts_then_drains_then_raises(tmp_path):
    created: list[FakeClient] = []
    context = FakeContext(workspace_path=tmp_path)

    async def request_cancel(client):
        context.control.put_nowait(InputResolved(input_id="unrelated"))
        context.control.put_nowait(CancelRequested())
        while not client.interrupted:
            await asyncio.sleep(0)

    backend = ClaudeAgentBackend(
        factory([text("a"), request_cancel, result(terminal_reason="aborted_streaming")], created)
    )
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(backend.execute(TurnOperation(prompt="x"), context), timeout=1)
    assert created[0].interrupted
    assert created[0].drained
    assert created[0].closed


async def test_cancel_wins_when_stream_and_control_complete_together(tmp_path):
    created: list[FakeClient] = []
    context = FakeContext(workspace_path=tmp_path)
    context.control.put_nowait(CancelRequested())
    backend = ClaudeAgentBackend(factory([result()], created))

    with pytest.raises(asyncio.CancelledError):
        await backend.execute(TurnOperation(prompt="x"), context)
    assert created[0].interrupted
    assert created[0].drained


async def test_outer_cancellation_waits_for_child_cleanup_before_client_exit(tmp_path):
    started = asyncio.Event()
    control_started = asyncio.Event()
    finish = asyncio.Event()
    order: list[str] = []

    class OrderingContext(FakeContext):
        async def wait_for_control(self):
            control_started.set()
            try:
                return await super().wait_for_control()
            finally:
                order.append("control_finished")

    class OrderingClient(FakeClient):
        async def receive_response(self):
            started.set()
            try:
                await finish.wait()
                yield result()
            finally:
                order.append("drain_finished")

        async def __aexit__(self, *_exc: object) -> bool:
            order.append("client_exited")
            return await super().__aexit__(*_exc)

    backend = ClaudeAgentBackend(lambda options: OrderingClient(options, []))
    task = asyncio.create_task(
        backend.execute(TurnOperation(prompt="x"), OrderingContext(workspace_path=tmp_path))
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.wait_for(control_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert set(order) == {"drain_finished", "control_finished", "client_exited"}
    assert order[-1] == "client_exited"


def _ask(tool, tool_input, outcomes, **permission):
    async def step(client):
        outcomes.append(
            await client.options.can_use_tool(
                tool, tool_input, ToolPermissionContext(tool_use_id="t1", **permission)
            )
        )

    return step


def _write_config(approval):
    return ResolvedExecutionConfig(sandbox="workspace_write", approval_policy=approval)


async def test_permission_ask_grant_carries_no_tool_body(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    outcomes: list = []
    secret = "TOP-SECRET-BODY"
    step = _ask(
        "Write",
        {"file_path": "/etc/outside.txt", "content": secret},
        outcomes,
        title="Claude wants to write /etc/outside.txt",
    )
    context = FakeContext(workspace_path=tmp_path, resolved_config=_write_config("on_request"))
    context.input_response = PermissionResponse(granted=frozenset({"Write:/etc/outside.txt"}))
    await ClaudeAgentBackend(factory([step, result()], [])).execute(
        TurnOperation(prompt="x"), context
    )
    request = context.input_requests[0]
    assert isinstance(request, PermissionRequest)
    assert request.prompt == "Claude wants to use Write"
    assert request.requested == frozenset({"Write:/etc/outside.txt"})
    assert secret not in request.model_dump_json()
    assert isinstance(outcomes[0], PermissionResultAllow)


async def test_permission_request_does_not_persist_opaque_sdk_context(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    secret = "TOP-SECRET-BODY"
    context = FakeContext(workspace_path=tmp_path, resolved_config=_write_config("on_request"))
    context.input_response = PermissionResponse()
    await ClaudeAgentBackend(
        factory(
            [
                _ask(
                    "Write",
                    {"file_path": "/etc/outside.txt", "content": secret},
                    [],
                    title=f"write {secret}",
                    blocked_path=f"blocked {secret}",
                    decision_reason=f"reason {secret}",
                ),
                result(),
            ],
            [],
        )
    ).execute(TurnOperation(prompt="x"), context)
    [request] = context.input_requests
    assert request.prompt == "Claude wants to use Write"
    assert request.risk == "Nexus sandbox policy requires approval"
    assert request.requested == frozenset({"Write:/etc/outside.txt"})
    assert secret not in request.model_dump_json()


async def test_permission_scope_too_long_is_denied_without_request(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    context = FakeContext(workspace_path=tmp_path, resolved_config=_write_config("on_request"))
    outcomes: list = []
    outside = "/etc/" + "/".join("x" * 100 for _ in range(25))
    context.input_response = PermissionResponse(granted=frozenset({f"Write:{outside}"[:2048]}))
    await ClaudeAgentBackend(
        factory([_ask("Write", {"file_path": outside}, outcomes), result()], [])
    ).execute(TurnOperation(prompt="x"), context)
    assert context.input_requests == []
    assert isinstance(outcomes[0], PermissionResultDeny)


async def test_notebook_permission_scopes_actual_notebook_path(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    outcomes: list = []
    inside = str(tmp_path / "inside.py")
    outside = "/etc/outside.ipynb"
    context = FakeContext(workspace_path=tmp_path, resolved_config=_write_config("on_request"))
    context.input_response = PermissionResponse(granted=frozenset({f"NotebookEdit:{outside}"}))
    await ClaudeAgentBackend(
        factory(
            [
                _ask(
                    "NotebookEdit",
                    {"file_path": inside, "notebook_path": outside, "new_source": "SECRET"},
                    outcomes,
                ),
                result(),
            ],
            [],
        )
    ).execute(TurnOperation(prompt="x"), context)
    [request] = context.input_requests
    assert request.requested == frozenset({f"NotebookEdit:{outside}"})
    assert inside not in request.model_dump_json()
    assert "SECRET" not in request.model_dump_json()
    assert isinstance(outcomes[0], PermissionResultAllow)


async def test_permission_ask_empty_grant_denies(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    outcomes: list = []
    context = FakeContext(workspace_path=tmp_path, resolved_config=_write_config("on_request"))
    context.input_response = PermissionResponse(granted=frozenset())
    await ClaudeAgentBackend(
        factory([_ask("WebFetch", {"url": "https://x"}, outcomes), result()], [])
    ).execute(TurnOperation(prompt="x"), context)
    assert context.input_requests[0].requested == frozenset({"WebFetch"})
    assert isinstance(outcomes[0], PermissionResultDeny)


async def test_never_policy_denies_without_asking(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "linux")
    outcomes: list = []
    context = FakeContext(workspace_path=tmp_path, resolved_config=_write_config("never"))
    await ClaudeAgentBackend(
        factory([_ask("WebFetch", {"url": "https://x"}, outcomes), result()], [])
    ).execute(TurnOperation(prompt="x"), context)
    assert context.input_requests == []
    assert isinstance(outcomes[0], PermissionResultDeny)


async def test_workspace_write_refused_off_posix(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    created: list[FakeClient] = []
    context = FakeContext(workspace_path=tmp_path, resolved_config=_write_config("never"))
    with pytest.raises(BackendFailure) as raised:
        await ClaudeAgentBackend(factory([result()], created)).execute(
            TurnOperation(prompt="x"), context
        )
    assert raised.value.error.code == "backend_unavailable"
    assert created == []
