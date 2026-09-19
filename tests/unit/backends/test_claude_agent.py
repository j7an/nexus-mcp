"""Claude Agent backend turn and configuration behavior."""

import pytest

from nexus_mcp.backends.base import BackendFailure
from nexus_mcp.backends.claude_agent import ClaudeAgentBackend
from nexus_mcp.core import (
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
