# tests/e2e/test_mcp_protocol.py
"""E2E MCP protocol tests using FastMCP's in-process Client.

Tests the full MCP stack that unit/pipeline tests miss:
- Tool discovery via list_tools() → JSON-RPC
- JSON-RPC argument serialization round-trips
- FastMCP DI injection of Progress and Context
- task=True background task lifecycle (Docket memory://)
- Schema validation at the protocol boundary

Mock boundary: asyncio.create_subprocess_exec only.
All layers above run for real, including JSON-RPC dispatch.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from tests.fixtures import GEMINI_NOISY_STDOUT, create_mock_process

# ---------------------------------------------------------------------------
# JSON response builder
# ---------------------------------------------------------------------------


def _gemini_json(output: str, stats: dict | None = None) -> str:
    """Build a Gemini CLI JSON response string."""
    data: dict = {"response": output}
    if stats:
        data["stats"] = stats
    return json.dumps(data)


def _gemini_error_json(code: int, message: str, status: str) -> str:
    """Build a Gemini API error JSON string."""
    return json.dumps({"error": {"code": code, "message": message, "status": status}})


# ---------------------------------------------------------------------------
# Class 1: Tool discovery
# ---------------------------------------------------------------------------


class TestToolDiscovery:
    """Verify MCP tool registration via list_tools() JSON-RPC call."""

    async def test_list_tools_returns_all_three(self, mcp_client):
        """list_tools() returns exactly 3 registered tools."""
        tools = await mcp_client.list_tools()
        names = {t.name for t in tools}
        assert names == {"prompt", "batch_prompt", "list_agents"}

    async def test_prompt_schema_has_required_params(self, mcp_client):
        """prompt tool schema requires 'agent' and 'prompt' parameters."""
        tools = await mcp_client.list_tools()
        prompt_tool = next(t for t in tools if t.name == "prompt")
        schema = prompt_tool.inputSchema
        assert schema is not None
        required = schema.get("required", [])
        assert "agent" in required
        assert "prompt" in required

    async def test_batch_prompt_schema_has_tasks_array(self, mcp_client):
        """batch_prompt tool schema requires 'tasks' as an array parameter."""
        tools = await mcp_client.list_tools()
        batch_tool = next(t for t in tools if t.name == "batch_prompt")
        schema = batch_tool.inputSchema
        assert schema is not None
        assert "tasks" in schema.get("required", [])
        assert schema["properties"]["tasks"]["type"] == "array"

    async def test_all_tools_have_descriptions(self, mcp_client):
        """All registered tools have non-empty descriptions."""
        tools = await mcp_client.list_tools()
        for tool in tools:
            assert tool.description, f"Tool {tool.name!r} has no description"


# ---------------------------------------------------------------------------
# Class 2: list_agents protocol
# ---------------------------------------------------------------------------


class TestListAgentsProtocol:
    """Verify list_agents tool via the MCP protocol."""

    async def test_list_agents_returns_gemini(self, mcp_client):
        """call_tool('list_agents') returns ['gemini'] through JSON-RPC."""
        result = await mcp_client.call_tool("list_agents", {})
        assert result.is_error is False
        assert result.data == ["gemini"]


# ---------------------------------------------------------------------------
# Class 3: prompt tool protocol
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_subprocess")
class TestPromptProtocol:
    """Verify the prompt tool through the full MCP protocol stack.

    Mock boundary: asyncio.create_subprocess_exec.
    JSON-RPC serialization, FastMCP DI injection, and tool dispatch run for real.
    """

    async def test_success_returns_parsed_output(self, mock_subprocess, mcp_client):
        """Full success path: subprocess returns valid JSON → call_tool returns output text."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("hello from e2e"))

        result = await mcp_client.call_tool("prompt", {"agent": "gemini", "prompt": "say hello"})

        assert result.is_error is False
        assert result.data == "hello from e2e"

    async def test_task_true_lifecycle(self, mock_subprocess, mcp_client):
        """task=True returns a ToolTask; awaiting it resolves to the final output."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("task result"))

        task = await mcp_client.call_tool(
            "prompt", {"agent": "gemini", "prompt": "background task"}, task=True
        )
        result = await task

        assert result.is_error is False
        assert result.data == "task result"

    async def test_model_parameter_reaches_subprocess(self, mock_subprocess, mcp_client):
        """model parameter survives JSON-RPC round-trip and appears in subprocess args."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))

        await mcp_client.call_tool(
            "prompt",
            {"agent": "gemini", "prompt": "test", "model": "gemini-2.5-flash"},
        )

        args = list(mock_subprocess.call_args.args)
        assert "--model" in args
        assert "gemini-2.5-flash" in args

    async def test_yolo_mode_reaches_subprocess(self, mock_subprocess, mcp_client):
        """execution_mode='yolo' survives JSON-RPC and appears as --yolo flag."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))

        await mcp_client.call_tool(
            "prompt",
            {"agent": "gemini", "prompt": "test", "execution_mode": "yolo"},
        )

        args = list(mock_subprocess.call_args.args)
        assert "--yolo" in args

    async def test_noisy_stdout_parsed_through_protocol(self, mock_subprocess, mcp_client):
        """Node.js warnings before JSON are stripped correctly through the full protocol."""
        mock_subprocess.return_value = create_mock_process(stdout=GEMINI_NOISY_STDOUT, returncode=0)

        result = await mcp_client.call_tool("prompt", {"agent": "gemini", "prompt": "noisy test"})

        assert result.is_error is False
        assert result.data == "test output"

    async def test_429_retry_then_success(self, mock_subprocess, mcp_client, fast_retry_sleep):
        """HTTP 429 triggers retry through the full protocol; second attempt succeeds."""
        error_json = _gemini_error_json(429, "Resource exhausted", "RESOURCE_EXHAUSTED")
        mock_subprocess.side_effect = [
            create_mock_process(stdout=error_json, returncode=1),
            create_mock_process(stdout=_gemini_json("ok after retry")),
        ]

        result = await mcp_client.call_tool(
            "prompt",
            {"agent": "gemini", "prompt": "test", "max_retries": 2},
        )

        assert result.is_error is False
        assert result.data == "ok after retry"
        assert mock_subprocess.call_count == 2


# ---------------------------------------------------------------------------
# Class 4: batch_prompt protocol
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_subprocess")
class TestBatchPromptProtocol:
    """Verify the batch_prompt tool through the full MCP protocol stack."""

    async def test_two_tasks_both_succeed(self, mock_subprocess, mcp_client):
        """Two tasks both succeed; response has correct succeeded/failed counts."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"agent": "gemini", "prompt": "task 1"},
                    {"agent": "gemini", "prompt": "task 2"},
                ]
            },
        )

        assert result.is_error is False
        assert result.data.succeeded == 2
        assert result.data.failed == 0

    async def test_partial_failure(self, mock_subprocess, mcp_client):
        """One task succeeds, one fails with ParseError → succeeded=1, failed=1."""

        def side_effect(*args, **kwargs):
            cmd = list(args)
            prompt_text = cmd[cmd.index("-p") + 1]
            if "succeed" in prompt_text:
                return create_mock_process(stdout=_gemini_json("ok"))
            return create_mock_process(stdout="not valid json", returncode=0)

        mock_subprocess.side_effect = side_effect

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"agent": "gemini", "prompt": "please succeed"},
                    {"agent": "gemini", "prompt": "will fail"},
                ]
            },
        )

        assert result.is_error is False
        assert result.data.succeeded == 1
        assert result.data.failed == 1
        failed = next(r for r in result.data.results if r.error is not None)
        assert failed.error_type == "ParseError"

    async def test_auto_labels_are_unique(self, mock_subprocess, mcp_client):
        """Auto-assigned labels for same-agent tasks are unique (gemini, gemini-2)."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"agent": "gemini", "prompt": "first"},
                    {"agent": "gemini", "prompt": "second"},
                ]
            },
        )

        labels = {r.label for r in result.data.results}
        assert "gemini" in labels
        assert "gemini-2" in labels

    async def test_explicit_labels_survive_json_rpc(self, mock_subprocess, mcp_client):
        """Explicit task labels survive JSON-RPC serialization round-trip."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"agent": "gemini", "prompt": "first", "label": "my-task-a"},
                    {"agent": "gemini", "prompt": "second", "label": "my-task-b"},
                ]
            },
        )

        labels = {r.label for r in result.data.results}
        assert labels == {"my-task-a", "my-task-b"}

    async def test_task_true_docket_coercion(self, mock_subprocess, mcp_client):
        """batch_prompt with task=True handles dict→AgentTask coercion after Docket."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("docket ok"))

        task = await mcp_client.call_tool(
            "batch_prompt",
            {"tasks": [{"agent": "gemini", "prompt": "docket test"}]},
            task=True,
        )
        result = await task

        assert result.is_error is False
        assert result.data.succeeded == 1


# ---------------------------------------------------------------------------
# Class 5: Error handling protocol
# ---------------------------------------------------------------------------


class TestErrorHandlingProtocol:
    """Verify error propagation through the MCP protocol layer."""

    async def test_unknown_agent_raises_tool_error(self, mcp_client):
        """Unknown agent name → UnsupportedAgentError → ToolError through protocol."""
        with pytest.raises(ToolError, match="UnsupportedAgentError"):
            await mcp_client.call_tool("prompt", {"agent": "unknown_agent", "prompt": "test"})

    async def test_parse_error_raises_tool_error(self, mcp_client, mock_subprocess):
        """Invalid JSON stdout → ParseError → ToolError with [ParseError] prefix."""
        mock_subprocess.return_value = create_mock_process(stdout="not valid json", returncode=0)

        with pytest.raises(ToolError, match=r"\[ParseError\]"):
            await mcp_client.call_tool("prompt", {"agent": "gemini", "prompt": "test"})

    async def test_missing_agent_param_rejected_by_schema(self, mcp_client):
        """Missing required 'agent' param is rejected at the MCP protocol/schema level."""
        with pytest.raises(ToolError):
            await mcp_client.call_tool("prompt", {"prompt": "test"})
