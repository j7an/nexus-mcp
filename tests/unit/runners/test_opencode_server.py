# tests/unit/runners/test_opencode_server.py
"""Tests for OpenCodeServerRunner.

Mock boundary: httpx.AsyncClient — all HTTP calls mocked.
"""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nexus_mcp.runners.opencode_server import OpenCodeServerRunner
from tests.fixtures import make_prompt_request


def make_server_runner() -> OpenCodeServerRunner:
    """Create an OpenCodeServerRunner using autouse cli_detection_mocks."""
    return OpenCodeServerRunner()


class TestOpenCodeServerRunnerInit:
    """Test runner construction and ABC stubs."""

    def test_agent_name(self):
        runner = make_server_runner()
        assert runner.AGENT_NAME == "opencode_server"

    def test_build_command_raises(self):
        runner = make_server_runner()
        request = make_prompt_request(cli="opencode_server")
        with pytest.raises(NotImplementedError):
            runner.build_command(request)

    def test_parse_output_raises(self):
        runner = make_server_runner()
        with pytest.raises(NotImplementedError):
            runner.parse_output("stdout", "stderr")


def _mock_post_session(session_id: str = "ses_abc123") -> AsyncMock:
    """Build a mock httpx.Response for POST /session."""
    resp = AsyncMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"id": session_id}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_get_session_ok(session_id: str = "ses_abc123") -> AsyncMock:
    """Build a mock httpx.Response for GET /session/:id (200)."""
    resp = AsyncMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"id": session_id}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_get_session_404() -> AsyncMock:
    """Build a mock httpx.Response for GET /session/:id (404)."""
    resp = AsyncMock(spec=httpx.Response)
    resp.status_code = 404
    return resp


class TestSessionManagement:
    """Test label-to-session mapping."""

    async def test_creates_session_on_first_prompt(self):
        runner = make_server_runner()
        mock_resp = _mock_post_session("ses_new")
        runner._client.post = AsyncMock(return_value=mock_resp)

        session_id = await runner._resolve_session("my-task")

        assert session_id == "ses_new"
        runner._client.post.assert_called_once()
        call_url = runner._client.post.call_args[0][0]
        assert call_url == "/session"

    async def test_reuses_cached_session(self):
        runner = make_server_runner()
        runner._sessions["my-task"] = "ses_existing"
        mock_get = _mock_get_session_ok("ses_existing")
        runner._client.get = AsyncMock(return_value=mock_get)

        session_id = await runner._resolve_session("my-task")

        assert session_id == "ses_existing"
        # Should validate, not create
        runner._client.get.assert_called_once()

    async def test_evicts_stale_session_and_creates_new(self):
        runner = make_server_runner()
        runner._sessions["my-task"] = "ses_stale"
        mock_get_404 = _mock_get_session_404()
        mock_post = _mock_post_session("ses_fresh")
        runner._client.get = AsyncMock(return_value=mock_get_404)
        runner._client.post = AsyncMock(return_value=mock_post)

        session_id = await runner._resolve_session("my-task")

        assert session_id == "ses_fresh"
        assert runner._sessions["my-task"] == "ses_fresh"

    async def test_ephemeral_session_no_caching(self):
        runner = make_server_runner()
        mock_post = _mock_post_session("ses_ephemeral")
        runner._client.post = AsyncMock(return_value=mock_post)

        session_id = await runner._resolve_session(None)

        assert session_id == "ses_ephemeral"
        assert None not in runner._sessions

    async def test_server_error_on_validate_creates_new_session(self):
        """Non-200, non-404 on GET /session/:id is treated as session unavailable."""
        runner = make_server_runner()
        runner._sessions["my-task"] = "ses_broken"
        mock_get_500 = AsyncMock(spec=httpx.Response)
        mock_get_500.status_code = 500
        mock_post = _mock_post_session("ses_new")
        runner._client.get = AsyncMock(return_value=mock_get_500)
        runner._client.post = AsyncMock(return_value=mock_post)

        session_id = await runner._resolve_session("my-task")

        assert session_id == "ses_new"


def _sse_lines(*events: str) -> str:
    """Build raw SSE text from data strings."""
    return "".join(f"data: {e}\n\n" for e in events)


def _mock_sse_response(sse_text: str) -> AsyncMock:
    """Build a mock streaming response that yields SSE lines."""
    resp = AsyncMock(spec=httpx.Response)
    resp.status_code = 200

    async def aiter_lines():
        for line in sse_text.splitlines(keepends=True):
            yield line

    resp.aiter_lines = aiter_lines
    return resp


class TestExecute:
    """Test _execute() flow: session -> prompt_async -> SSE -> AgentResponse."""

    async def test_basic_prompt_returns_response(self):
        runner = make_server_runner()

        # Mock session creation
        runner._resolve_session = AsyncMock(return_value="ses_test")

        # Mock prompt_async
        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204
        prompt_resp.raise_for_status = MagicMock()

        # Mock SSE stream with a text event and completion
        sse_data = json.dumps(
            {
                "type": "text",
                "sessionID": "ses_test",
                "part": {"type": "text", "text": "Hello world"},
            }
        )
        completion = json.dumps({"type": "step_finish", "sessionID": "ses_test"})
        sse_text = f"data: {sse_data}\n\ndata: {completion}\n\n"
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        request = make_prompt_request(cli="opencode_server", prompt="Say hello")
        response = await runner._execute(request)

        assert response.cli == "opencode_server"
        assert response.output == "Hello world"

    async def test_context_serialized_to_system_field(self):
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_test")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204
        prompt_resp.raise_for_status = MagicMock()

        sse_data = json.dumps(
            {"type": "text", "sessionID": "ses_test", "part": {"type": "text", "text": "ok"}}
        )
        completion = json.dumps({"type": "step_finish", "sessionID": "ses_test"})
        sse_text = f"data: {sse_data}\n\ndata: {completion}\n\n"
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        request = make_prompt_request(
            cli="opencode_server",
            prompt="Do thing",
            context={"project": "nexus-mcp"},
        )
        await runner._execute(request)

        # Verify prompt_async was called with system field
        post_call = runner._client.post.call_args
        body = post_call.kwargs.get("json") or post_call[1].get("json")
        assert "system" in body
        assert json.loads(body["system"]) == {"project": "nexus-mcp"}

    async def test_model_included_in_payload(self):
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_test")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204
        prompt_resp.raise_for_status = MagicMock()

        sse_data = json.dumps(
            {"type": "text", "sessionID": "ses_test", "part": {"type": "text", "text": "ok"}}
        )
        completion = json.dumps({"type": "step_finish", "sessionID": "ses_test"})
        sse_text = f"data: {sse_data}\n\ndata: {completion}\n\n"
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        request = make_prompt_request(cli="opencode_server", model="anthropic/claude-3-opus")
        await runner._execute(request)

        post_call = runner._client.post.call_args
        body = post_call.kwargs.get("json") or post_call[1].get("json")
        assert body["model"] == "anthropic/claude-3-opus"
