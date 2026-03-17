# tests/e2e/test_preferences_protocol.py
"""E2E MCP protocol tests for session preference tools.

Verifies the full stack: JSON-RPC serialization → FastMCP DI → preference tools →
session state persistence → prompt/batch_prompt preference resolution.
"""

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from nexus_mcp.server import mcp
from tests.fixtures import create_mock_process, gemini_json

# ---------------------------------------------------------------------------
# Class 1: set/get/clear round-trips
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPreferencesRoundTrip:
    async def test_set_then_get_returns_preference(self, mcp_client):
        """set_preferences → get_preferences returns the stored execution_mode."""
        await mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.is_error is False
        assert result.data["execution_mode"] == "yolo"

    async def test_get_returns_defaults_before_set(self, mcp_client):
        """get_preferences before any set_preferences returns None values."""
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.is_error is False
        assert result.data == {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": None,
            "timeout": None,
            "retry_base_delay": None,
            "retry_max_delay": None,
        }

    async def test_set_model_preference(self, mcp_client):
        """set_preferences(model=...) is returned by get_preferences."""
        await mcp_client.call_tool("set_preferences", {"model": "gemini-2.5-flash"})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data["model"] == "gemini-2.5-flash"

    async def test_set_merges_with_existing(self, mcp_client):
        """Two sequential set_preferences calls merge: first sets mode, second sets model."""
        await mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})
        await mcp_client.call_tool("set_preferences", {"model": "gemini-2.5-flash"})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data["execution_mode"] == "yolo"
        assert result.data["model"] == "gemini-2.5-flash"

    async def test_clear_resets_to_defaults(self, mcp_client):
        """clear_preferences → get_preferences returns None values."""
        await mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})
        await mcp_client.call_tool("clear_preferences", {})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data == {
            "execution_mode": None,
            "model": None,
            "max_retries": None,
            "output_limit": None,
            "timeout": None,
            "retry_base_delay": None,
            "retry_max_delay": None,
        }

    async def test_set_max_retries_preference(self, mcp_client):
        """set_preferences(max_retries=5) is returned by get_preferences."""
        await mcp_client.call_tool("set_preferences", {"max_retries": 5})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data["max_retries"] == 5

    async def test_set_output_limit_preference(self, mcp_client):
        """set_preferences(output_limit=4096) is returned by get_preferences."""
        await mcp_client.call_tool("set_preferences", {"output_limit": 4096})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data["output_limit"] == 4096

    async def test_set_timeout_preference(self, mcp_client):
        """set_preferences(timeout=30) is returned by get_preferences."""
        await mcp_client.call_tool("set_preferences", {"timeout": 30})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data["timeout"] == 30

    async def test_clear_individual_new_fields(self, mcp_client):
        """clear_max_retries=True clears max_retries while preserving other fields."""
        await mcp_client.call_tool("set_preferences", {"max_retries": 5, "timeout": 30})
        await mcp_client.call_tool("set_preferences", {"clear_max_retries": True})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data["max_retries"] is None
        assert result.data["timeout"] == 30  # preserved

    async def test_set_retry_delay_preferences(self, mcp_client):
        """set_preferences(retry_base_delay=1.5, retry_max_delay=60.0) stores both fields."""
        await mcp_client.call_tool(
            "set_preferences", {"retry_base_delay": 1.5, "retry_max_delay": 60.0}
        )
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data["retry_base_delay"] == 1.5
        assert result.data["retry_max_delay"] == 60.0

    async def test_clear_retry_base_delay_preference(self, mcp_client):
        """clear_retry_base_delay=True resets retry_base_delay to None."""
        await mcp_client.call_tool("set_preferences", {"retry_base_delay": 1.5})
        await mcp_client.call_tool("set_preferences", {"clear_retry_base_delay": True})
        result = await mcp_client.call_tool("get_preferences", {})
        assert result.data["retry_base_delay"] is None

    async def test_set_preferences_returns_confirmation(self, mcp_client):
        """set_preferences returns a non-empty confirmation string."""
        result = await mcp_client.call_tool("set_preferences", {"execution_mode": "default"})
        assert result.is_error is False
        assert isinstance(result.data, str)
        assert len(result.data) > 0

    async def test_clear_preferences_returns_confirmation(self, mcp_client):
        """clear_preferences returns a non-empty confirmation string."""
        result = await mcp_client.call_tool("clear_preferences", {})
        assert result.is_error is False
        assert "cleared" in result.data.lower()


# ---------------------------------------------------------------------------
# Class 2: preference → prompt interaction
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPreferenceAffectsPrompt:
    async def test_session_execution_mode_reaches_subprocess(self, mock_subprocess, mcp_client):
        """Session execution_mode='yolo' causes --yolo flag in subprocess args."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})
        await mcp_client.call_tool("prompt", {"cli": "gemini", "prompt": "test"})

        args = list(mock_subprocess.call_args.args)
        assert "--yolo" in args

    async def test_explicit_mode_overrides_session_in_subprocess(self, mock_subprocess, mcp_client):
        """Explicit execution_mode='default' overrides session 'yolo' in subprocess args."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})
        await mcp_client.call_tool(
            "prompt", {"cli": "gemini", "prompt": "test", "execution_mode": "default"}
        )

        args = list(mock_subprocess.call_args.args)
        assert "--yolo" not in args

    async def test_session_model_reaches_subprocess(self, mock_subprocess, mcp_client):
        """Session model appears as --model flag in subprocess args."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool("set_preferences", {"model": "gemini-2.5-flash"})
        await mcp_client.call_tool("prompt", {"cli": "gemini", "prompt": "test"})

        args = list(mock_subprocess.call_args.args)
        assert "--model" in args
        assert "gemini-2.5-flash" in args

    async def test_explicit_model_overrides_session_model(self, mock_subprocess, mcp_client):
        """Explicit model='gemini-1.5-pro' overrides session 'gemini-2.5-flash'."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool("set_preferences", {"model": "gemini-2.5-flash"})
        await mcp_client.call_tool(
            "prompt", {"cli": "gemini", "prompt": "test", "model": "gemini-1.5-pro"}
        )

        args = list(mock_subprocess.call_args.args)
        assert "gemini-1.5-pro" in args
        assert "gemini-2.5-flash" not in args

    async def test_no_mode_without_session_uses_default(self, mock_subprocess, mcp_client):
        """Without session or explicit mode, prompt uses 'default' (no --yolo flag)."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool("prompt", {"cli": "gemini", "prompt": "test"})

        args = list(mock_subprocess.call_args.args)
        assert "--yolo" not in args


# ---------------------------------------------------------------------------
# Class 3: preference → batch_prompt interaction
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPreferenceAffectsBatchPrompt:
    async def test_session_mode_applied_to_task_without_mode(self, mock_subprocess, mcp_client):
        """Session execution_mode='yolo' applies to tasks that don't specify execution_mode."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})
        result = await mcp_client.call_tool(
            "batch_prompt",
            {"tasks": [{"cli": "gemini", "prompt": "test"}]},
        )

        assert result.is_error is False
        assert result.data.succeeded == 1
        # --yolo appeared in subprocess call
        args = list(mock_subprocess.call_args.args)
        assert "--yolo" in args

    async def test_task_explicit_mode_not_overridden_by_session(self, mock_subprocess, mcp_client):
        """Task with explicit execution_mode='default' keeps it despite session 'yolo'."""
        mock_subprocess.return_value = create_mock_process(stdout=gemini_json("ok"))

        await mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})
        await mcp_client.call_tool(
            "batch_prompt",
            {"tasks": [{"cli": "gemini", "prompt": "test", "execution_mode": "default"}]},
        )

        args = list(mock_subprocess.call_args.args)
        assert "--yolo" not in args


# ---------------------------------------------------------------------------
# Class 4: Input validation at protocol boundary
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestPreferencesValidation:
    async def test_invalid_execution_mode_raises_tool_error(self, mcp_client):
        """set_preferences with an unrecognised execution_mode is rejected by Pydantic."""
        with pytest.raises(ToolError, match="execution_mode"):
            await mcp_client.call_tool("set_preferences", {"execution_mode": "invalid_value"})

    async def test_empty_model_string_raises_tool_error(self, mcp_client):
        """set_preferences with model='' is rejected (min_length=1 on SessionPreferences.model)."""
        with pytest.raises(ToolError):
            await mcp_client.call_tool("set_preferences", {"model": ""})

    async def test_valid_preferences_after_rejected_call(self, mcp_client):
        """A rejected call does not corrupt session state; valid call succeeds afterwards."""
        with pytest.raises(ToolError):
            await mcp_client.call_tool("set_preferences", {"execution_mode": "invalid_value"})
        result = await mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})
        assert result.is_error is False
        prefs = await mcp_client.call_tool("get_preferences", {})
        assert prefs.data["execution_mode"] == "yolo"


# ---------------------------------------------------------------------------
# Class 5: Session isolation
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSessionIsolation:
    async def test_new_session_does_not_see_previous_session_preferences(self):
        """Two sequential sessions are isolated: second session has no preferences."""
        # Session 1: set yolo preference
        async with Client(mcp) as client1:
            await client1.call_tool("set_preferences", {"execution_mode": "yolo"})
            r1 = await client1.call_tool("get_preferences", {})
            assert r1.data["execution_mode"] == "yolo"
        mcp._lifespan_result_set = False  # FastMCP bug workaround

        # Session 2: no preferences set — should see defaults
        async with Client(mcp) as client2:
            r2 = await client2.call_tool("get_preferences", {})
            assert r2.data["execution_mode"] is None
        mcp._lifespan_result_set = False
