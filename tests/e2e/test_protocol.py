"""End-to-end coverage for the MCP protocol surface."""

import json

import pytest

pytestmark = pytest.mark.e2e


async def test_lists_exactly_the_prompt_tool(client):
    tools = await client.list_tools()
    assert [tool.name for tool in tools] == ["prompt"]
    [tool] = tools
    assert tool.annotations.destructive_hint is True
    assert tool.annotations.read_only_hint is False
    assert set(tool.input_schema["required"]) == {"backend", "prompt", "cwd"}
    assert tool.output_schema is not None


async def test_prompt_round_trip(client, fake_backend, tmp_path):
    result = await client.call_tool(
        "prompt", {"backend": "claude", "prompt": "hello", "cwd": str(tmp_path)}
    )
    assert result.structured_content == {
        "backend": "claude",
        "session_id": "sid-1",
        "output": "echo: hello",
        "usage": {"n": 1},
    }
    assert fake_backend.requests[0].profile == "read_only"


async def test_validation_error_is_tool_error(client, tmp_path):
    result = await client.call_tool(
        "prompt",
        {"backend": "claude", "prompt": "hi", "cwd": "relative"},
        raise_on_error=False,
    )
    assert result.is_error is True
    assert "cwd" in result.content[0].text


async def test_unknown_profile_rejected_by_schema(client, tmp_path):
    result = await client.call_tool(
        "prompt",
        {"backend": "claude", "prompt": "hi", "cwd": str(tmp_path), "profile": "yolo"},
        raise_on_error=False,
    )
    assert result.is_error is True


async def test_backends_resource(client):
    contents = await client.read_resource("nexus://backends")
    assert json.loads(contents[0].text) == [
        {"name": "claude", "installed": True, "models": None, "hint": None}
    ]
