"""Tests for session-related OpenCode MCP resources."""

import httpx
import pytest
import respx

from nexus_mcp.http_client import reset_http_client


@pytest.fixture(autouse=True)
def _mock_cli(mock_cli_detection):
    yield mock_cli_detection


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch):
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://test:4096")
    monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "testpass")
    reset_http_client()
    yield
    reset_http_client()


class TestSessionsResource:
    @respx.mock
    async def test_returns_session_list(self):
        respx.get("http://test:4096/session").mock(
            return_value=httpx.Response(
                200,
                json=[{"id": "ses_abc", "status": "idle"}, {"id": "ses_def", "status": "active"}],
            )
        )
        from nexus_mcp.opencode_resources import get_opencode_sessions

        result = await get_opencode_sessions()
        assert "ses_abc" in result
        assert "ses_def" in result

    @respx.mock
    async def test_propagates_http_error(self):
        respx.get("http://test:4096/session").mock(return_value=httpx.Response(500))
        from nexus_mcp.exceptions import SubprocessError
        from nexus_mcp.opencode_resources import get_opencode_sessions

        with pytest.raises(SubprocessError):
            await get_opencode_sessions()


class TestSessionsStatusResource:
    @respx.mock
    async def test_returns_status_map(self):
        respx.get("http://test:4096/session/status").mock(
            return_value=httpx.Response(200, json={"ses_abc": "idle", "ses_def": "active"})
        )
        from nexus_mcp.opencode_resources import get_opencode_sessions_status

        result = await get_opencode_sessions_status()
        assert "idle" in result
        assert "active" in result


class TestPermissionsResource:
    @respx.mock
    async def test_returns_pending_permissions(self):
        respx.get("http://test:4096/permission").mock(
            return_value=httpx.Response(
                200,
                json=[{"id": "per_123", "sessionID": "ses_abc", "permission": "bash"}],
            )
        )
        from nexus_mcp.opencode_resources import get_opencode_permissions

        result = await get_opencode_permissions()
        assert "per_123" in result
        assert "bash" in result

    @respx.mock
    async def test_returns_empty_list_when_none_pending(self):
        respx.get("http://test:4096/permission").mock(return_value=httpx.Response(200, json=[]))
        from nexus_mcp.opencode_resources import get_opencode_permissions

        result = await get_opencode_permissions()
        assert "[]" in result


class TestQuestionsResource:
    @respx.mock
    async def test_returns_pending_questions(self):
        respx.get("http://test:4096/question").mock(
            return_value=httpx.Response(
                200,
                json=[{"id": "que_456", "sessionID": "ses_abc", "questions": []}],
            )
        )
        from nexus_mcp.opencode_resources import get_opencode_questions

        result = await get_opencode_questions()
        assert "que_456" in result
