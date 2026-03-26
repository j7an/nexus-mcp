# tests/e2e/test_mcp_protocol.py
"""E2E MCP protocol tests using FastMCP's in-process Client.

Tests the full MCP stack that unit/pipeline tests miss:
- Tool discovery via list_tools() → JSON-RPC
- JSON-RPC argument serialization round-trips
- FastMCP DI injection of Context
- task=True background task lifecycle (Docket memory://)
- Schema validation at the protocol boundary

Mock boundary: asyncio.create_subprocess_exec only.
All layers above run for real, including JSON-RPC dispatch.
"""

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.server import mcp
from tests.fixtures import (
    GEMINI_NOISY_STDOUT,
    OPENCODE_NDJSON_RESPONSE,
    create_mock_process,
    gemini_error_json,
    gemini_json,
    strip_runner_header,
)


def _extract_prompt_from_args(args: tuple) -> str:
    """Extract prompt from subprocess args.

    Mirrors GeminiRunner.build_command argument layout: [..., -p, <prompt>, ...].
    """
    cmd = list(args)
    return cmd[cmd.index("-p") + 1]


# ---------------------------------------------------------------------------
# Class 1: Tool discovery
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestToolDiscovery:
    """Verify MCP tool registration via list_tools() JSON-RPC call."""

    async def test_list_tools_returns_all_five(self, mcp_client):
        """list_tools() returns exactly 5 registered tools."""
        tools = await mcp_client.list_tools()
        names = {t.name for t in tools}
        assert names == {
            "prompt",
            "batch_prompt",
            "set_preferences",
            "get_preferences",
            "clear_preferences",
        }

    async def test_prompt_schema_has_required_params(self, mcp_client):
        """prompt tool schema requires 'prompt'; 'cli' is optional (elicitation)."""
        tools = await mcp_client.list_tools()
        prompt_tool = next(t for t in tools if t.name == "prompt")
        schema = prompt_tool.inputSchema
        assert schema is not None
        required = schema.get("required", [])
        assert "prompt" in required
        # cli is optional — can be resolved via elicitation
        assert "cli" not in required
        assert "cli" in schema.get("properties", {})

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
# Class 2: Tool annotations
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestToolAnnotations:
    """Verify MCP tool annotations are set correctly on all tools."""

    async def test_execution_tools_have_destructive_open_world_annotations(self, mcp_client):
        """prompt and batch_prompt are destructive + open-world (they shell out to CLIs)."""
        tools = await mcp_client.list_tools()
        for name in ("prompt", "batch_prompt"):
            tool = next(t for t in tools if t.name == name)
            assert tool.annotations is not None, f"{name} missing annotations"
            assert tool.annotations.readOnlyHint is False
            assert tool.annotations.destructiveHint is True
            assert tool.annotations.idempotentHint is False
            assert tool.annotations.openWorldHint is True

    async def test_get_preferences_is_read_only_and_idempotent(self, mcp_client):
        """get_preferences is a pure read with no side effects."""
        tools = await mcp_client.list_tools()
        tool = next(t for t in tools if t.name == "get_preferences")
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False

    async def test_set_preferences_is_idempotent_non_destructive(self, mcp_client):
        """set_preferences merges state (non-destructive) and is idempotent."""
        tools = await mcp_client.list_tools()
        tool = next(t for t in tools if t.name == "set_preferences")
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False
        assert tool.annotations.destructiveHint is False
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False

    async def test_clear_preferences_is_destructive_and_idempotent(self, mcp_client):
        """clear_preferences erases all state (destructive) but clearing twice is the same."""
        tools = await mcp_client.list_tools()
        tool = next(t for t in tools if t.name == "clear_preferences")
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False
        assert tool.annotations.destructiveHint is True
        assert tool.annotations.idempotentHint is True
        assert tool.annotations.openWorldHint is False

    async def test_all_tools_have_titles(self, mcp_client):
        """Every tool has a human-readable title set via annotations."""
        expected_titles = {
            "prompt": "Prompt CLI Agent",
            "batch_prompt": "Batch Prompt CLI Agents",
            "set_preferences": "Set Session Preferences",
            "get_preferences": "Get Session Preferences",
            "clear_preferences": "Clear Session Preferences",
        }
        tools = await mcp_client.list_tools()
        for tool in tools:
            assert tool.annotations is not None, f"{tool.name} missing annotations"
            assert tool.annotations.title == expected_titles[tool.name], (
                f"{tool.name}: expected title {expected_titles[tool.name]!r}, "
                f"got {tool.annotations.title!r}"
            )


# ---------------------------------------------------------------------------
# Class 3: Server instructions protocol
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestServerInstructionsProtocol:
    """Verify server instructions are delivered via MCP protocol."""

    async def test_instructions_contain_runner_names(self, mcp_client):
        """Server instructions mention all registered runner names."""
        # FastMCP exposes instructions via the server info
        from nexus_mcp.server import mcp

        assert mcp.instructions is not None
        for name in ("claude", "codex", "gemini", "opencode"):
            assert name in mcp.instructions


# ---------------------------------------------------------------------------
# Class 3: prompt tool protocol
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPromptProtocol:
    """Verify the prompt tool through the full MCP protocol stack.

    Mock boundary: asyncio.create_subprocess_exec.
    JSON-RPC serialization, FastMCP DI injection, and tool dispatch run for real.
    """

    async def test_success_returns_parsed_output(self, mock_subprocess, mcp_client):
        """Full success path: subprocess returns valid JSON → call_tool returns output text."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("hello from e2e"))

        result = await mcp_client.call_tool("prompt", {"cli": "gemini", "prompt": "say hello"})

        assert result.is_error is False
        assert strip_runner_header(result.data) == "hello from e2e"

    async def test_task_true_lifecycle(self, mock_subprocess, mcp_client):
        """task=True returns a ToolTask; awaiting it resolves to the final output."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("task result"))

        task = await mcp_client.call_tool(
            "prompt", {"cli": "gemini", "prompt": "background task"}, task=True
        )
        result = await task

        assert result.is_error is False
        assert strip_runner_header(result.data) == "task result"

    async def test_model_parameter_reaches_subprocess(self, mock_subprocess, mcp_client):
        """model parameter survives JSON-RPC round-trip and appears in subprocess args."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool(
            "prompt",
            {"cli": "gemini", "prompt": "test", "model": "gemini-2.5-flash"},
        )

        args = list(mock_subprocess.call_args.args)
        assert "--model" in args
        assert "gemini-2.5-flash" in args

    async def test_yolo_mode_reaches_subprocess(self, mock_subprocess, mcp_client):
        """execution_mode='yolo' survives JSON-RPC and appears as --yolo flag."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool(
            "prompt",
            {"cli": "gemini", "prompt": "test", "execution_mode": "yolo"},
        )

        args = list(mock_subprocess.call_args.args)
        assert "--yolo" in args

    async def test_noisy_stdout_parsed_through_protocol(self, mock_subprocess, mcp_client):
        """Node.js warnings before JSON are stripped correctly through the full protocol."""
        mock_subprocess.return_value = create_mock_process(stdout=GEMINI_NOISY_STDOUT, returncode=0)

        result = await mcp_client.call_tool("prompt", {"cli": "gemini", "prompt": "noisy test"})

        assert result.is_error is False
        assert strip_runner_header(result.data) == "test output"

    @pytest.mark.parametrize(
        ("error_code", "error_message", "error_status"),
        [
            (429, "Resource exhausted", "RESOURCE_EXHAUSTED"),
            (503, "Service unavailable", "UNAVAILABLE"),
        ],
        ids=["429-rate-limit", "503-unavailable"],
    )
    async def test_retryable_error_retry_then_success(
        self,
        mock_subprocess,
        mcp_client,
        fast_retry_sleep,
        error_code,
        error_message,
        error_status,
    ):
        """Retryable HTTP error triggers retry; second attempt succeeds."""
        error_json = gemini_error_json(error_code, error_message, error_status)
        mock_subprocess.side_effect = [
            create_mock_process(stdout=error_json, returncode=1),
            create_mock_process(stdout=gemini_json("ok after retry")),
        ]

        result = await mcp_client.call_tool(
            "prompt",
            {"cli": "gemini", "prompt": "test", "max_retries": 2},
        )

        assert result.is_error is False
        assert strip_runner_header(result.data) == "ok after retry"
        assert mock_subprocess.call_count == 2

    async def test_opencode_success_returns_parsed_ndjson(self, mock_subprocess, mcp_client):
        """Full success path for OpenCode: subprocess returns NDJSON → parsed output text."""
        mock_subprocess.return_value = create_mock_process(stdout=OPENCODE_NDJSON_RESPONSE)
        result = await mcp_client.call_tool("prompt", {"cli": "opencode", "prompt": "say hello"})
        assert result.is_error is False
        assert strip_runner_header(result.data) == "test output"

    async def test_context_parameter_survives_json_rpc(self, mock_subprocess, mcp_client):
        """context dict survives JSON-RPC round-trip; call succeeds (context is pass-through)."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        result = await mcp_client.call_tool(
            "prompt",
            {
                "cli": "gemini",
                "prompt": "test",
                "context": {"session_id": "abc-123", "metadata": {"nested": True}},
            },
        )

        assert result.is_error is False
        assert strip_runner_header(result.data) == "ok"


# ---------------------------------------------------------------------------
# Class 4: batch_prompt protocol
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestBatchPromptProtocol:
    """Verify the batch_prompt tool through the full MCP protocol stack."""

    async def test_two_tasks_both_succeed(self, mock_subprocess, mcp_client):
        """Two tasks both succeed; response has correct succeeded/failed counts."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "gemini", "prompt": "task 1"},
                    {"cli": "gemini", "prompt": "task 2"},
                ]
            },
        )

        assert result.is_error is False
        assert result.data.succeeded == 2
        assert result.data.failed == 0

    async def test_partial_failure(self, mock_subprocess, mcp_client):
        """One task succeeds, one fails with ParseError → succeeded=1, failed=1."""

        def side_effect(*args, **kwargs):
            prompt_text = _extract_prompt_from_args(args)
            if "succeed" in prompt_text:
                return create_mock_process(stdout=gemini_json("ok"))
            return create_mock_process(stdout="not valid json", returncode=0)

        mock_subprocess.side_effect = side_effect

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "gemini", "prompt": "please succeed"},
                    {"cli": "gemini", "prompt": "will fail"},
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
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "gemini", "prompt": "first"},
                    {"cli": "gemini", "prompt": "second"},
                ]
            },
        )

        labels = {r.label for r in result.data.results}
        assert "gemini" in labels
        assert "gemini-2" in labels

    async def test_explicit_labels_survive_json_rpc(self, mock_subprocess, mcp_client):
        """Explicit task labels survive JSON-RPC serialization round-trip."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "gemini", "prompt": "first", "label": "my-task-a"},
                    {"cli": "gemini", "prompt": "second", "label": "my-task-b"},
                ]
            },
        )

        labels = {r.label for r in result.data.results}
        assert labels == {"my-task-a", "my-task-b"}

    async def test_task_true_docket_coercion(self, mock_subprocess, mcp_client):
        """batch_prompt with task=True handles dict→AgentTask coercion after Docket."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("docket ok"))

        task = await mcp_client.call_tool(
            "batch_prompt",
            {"tasks": [{"cli": "gemini", "prompt": "docket test"}]},
            task=True,
        )
        result = await task

        assert result.is_error is False
        assert result.data.succeeded == 1

    async def test_max_concurrency_parameter_accepted(self, mock_subprocess, mcp_client):
        """max_concurrency=1 is accepted through JSON-RPC without error; both tasks succeed."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        result = await mcp_client.call_tool(
            "batch_prompt",
            {
                "tasks": [
                    {"cli": "gemini", "prompt": "task 1"},
                    {"cli": "gemini", "prompt": "task 2"},
                ],
                "max_concurrency": 1,
            },
        )

        assert result.is_error is False
        assert result.data.succeeded == 2
        assert result.data.failed == 0

    async def test_empty_task_list_returns_empty_response(self, mcp_client):
        """batch_prompt with tasks=[] returns empty MultiPromptResponse (total=0, results=[])."""
        result = await mcp_client.call_tool("batch_prompt", {"tasks": []})

        assert result.is_error is False
        assert result.data.total == 0
        assert result.data.results == []


# ---------------------------------------------------------------------------
# Class 5: Tool timeout
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestToolTimeout:
    """Verify FastMCP tool-level timeout via anyio.fail_after().

    Tools registered with task=True support both synchronous and background
    dispatch. The timeout applies on the synchronous path (client calls
    without task=True). Background calls go through Docket and are protected
    by the subprocess-level timeout instead.
    """

    async def test_hung_tool_times_out(self, mock_subprocess, mcp_client, monkeypatch):
        """A hung subprocess is cancelled by the tool-level anyio.fail_after().

        Patching tool.timeout to 0.5s and simulating a 5s subprocess delay
        triggers a TimeoutError on the server side. FastMCP converts this to
        an isError=True result, which the client re-raises as ToolError with
        a "timed out after" message.
        """
        tool = await mcp.get_tool("prompt")
        monkeypatch.setattr(tool, "timeout", 0.5)
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"), delay=5.0)
        with pytest.raises(ToolError, match="timed out after 0\\.5s"):
            await mcp_client.call_tool("prompt", {"cli": "gemini", "prompt": "test"})


# ---------------------------------------------------------------------------
# Class 6: Error handling protocol
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestErrorHandlingProtocol:
    """Verify error propagation through the MCP protocol layer."""

    async def test_unknown_agent_raises_tool_error(self, mcp_client):
        """Unknown agent name → UnsupportedAgentError → ToolError through protocol."""
        with pytest.raises(ToolError, match="UnsupportedAgentError"):
            await mcp_client.call_tool("prompt", {"cli": "unknown_agent", "prompt": "test"})

    async def test_parse_error_raises_tool_error(self, mcp_client, mock_subprocess):
        """Invalid JSON stdout → ParseError → ToolError with [ParseError] prefix."""
        mock_subprocess.return_value = create_mock_process(stdout="not valid json", returncode=0)

        with pytest.raises(ToolError, match=r"\[ParseError\]"):
            await mcp_client.call_tool("prompt", {"cli": "gemini", "prompt": "test"})

    async def test_missing_cli_param_rejected_by_schema(self, mcp_client):
        """Missing required 'cli' param is rejected at the MCP protocol/schema level."""
        with pytest.raises(ToolError):
            await mcp_client.call_tool("prompt", {"prompt": "test"})

    async def test_non_retryable_error_raises_tool_error(self, mcp_client, mock_subprocess):
        """HTTP 401 (non-retryable) raises ToolError immediately with no retry attempts."""
        error_json = gemini_error_json(401, "API key not valid", "UNAUTHENTICATED")
        mock_subprocess.return_value = create_mock_process(stdout=error_json, returncode=1)

        with pytest.raises(ToolError, match="SubprocessError"):
            await mcp_client.call_tool(
                "prompt",
                {"cli": "gemini", "prompt": "test", "max_retries": 3},
            )

        assert mock_subprocess.call_count == 1
