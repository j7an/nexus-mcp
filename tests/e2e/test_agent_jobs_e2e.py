"""End-to-end MCP protocol contracts for durable agent jobs."""

import asyncio

import pytest


@pytest.mark.e2e
async def test_agent_start_requires_workspace(mcp_client):
    """A job cannot inherit an ambient server working directory."""
    result = await mcp_client.call_tool(
        "agent_start",
        {"backend": "fake", "prompt": "x"},
        raise_on_error=False,
    )

    assert result.is_error is True
    assert "workspace" in result.content[0].text.casefold()


@pytest.mark.e2e
async def test_agent_start_status_and_result_round_trip(job_mcp_client, tmp_path):
    """A queued fake turn reaches a typed terminal result through public tools."""
    started = await job_mcp_client.call_tool(
        "agent_start",
        {
            "workspace": {"path": str(tmp_path)},
            "backend": "fake",
            "prompt": "return a durable result",
            "context": {"fake_output": "durable fake output"},
        },
    )
    handle = started.structured_content
    assert handle is not None
    assert handle["state"] == "queued"
    assert handle["session_id"]

    for _ in range(500):
        current = await job_mcp_client.call_tool(
            "agent_status",
            {"workspace": {"path": str(tmp_path)}, "job_id": handle["job_id"]},
        )
        status = current.structured_content
        assert status is not None
        if status["state"] in {"completed", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("durable fake job did not reach a terminal state")

    assert status["state"] == "completed"
    completed = await job_mcp_client.call_tool(
        "agent_result",
        {"workspace": {"path": str(tmp_path)}, "job_id": handle["job_id"]},
    )
    result = completed.data
    assert isinstance(result, dict)
    assert result["status"] == "succeeded"
    assert result["result"]["payload"] == {
        "kind": "turn",
        "message": "durable fake output",
        "structured_output": None,
        "changed_files": [],
        "commands": [],
        "usage": {},
    }
