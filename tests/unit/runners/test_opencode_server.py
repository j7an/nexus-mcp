# tests/unit/runners/test_opencode_server.py
"""Tests for OpenCodeServerRunner.

Mock boundary: httpx.AsyncClient — all HTTP calls mocked.
"""

from unittest.mock import AsyncMock

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
    resp.raise_for_status = AsyncMock()
    return resp


def _mock_get_session_ok(session_id: str = "ses_abc123") -> AsyncMock:
    """Build a mock httpx.Response for GET /session/:id (200)."""
    resp = AsyncMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"id": session_id}
    resp.raise_for_status = AsyncMock()
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
