"""Tests for session-related OpenCode MCP resources."""

import httpx
import pytest
import respx
from fastmcp.exceptions import ResourceError

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


class TestSessionIdValidation:
    """Test ID validation patterns used by URI template resources."""

    def test_valid_session_id_passes(self):
        from nexus_mcp.opencode_resources import _SESSION_ID_PATTERN

        assert _SESSION_ID_PATTERN.fullmatch("ses_abc123")
        assert _SESSION_ID_PATTERN.fullmatch("ses-def-456")
        assert _SESSION_ID_PATTERN.fullmatch("sesABC")

    def test_invalid_session_id_fails(self):
        from nexus_mcp.opencode_resources import _SESSION_ID_PATTERN

        assert _SESSION_ID_PATTERN.fullmatch("../../etc/passwd") is None
        assert _SESSION_ID_PATTERN.fullmatch("") is None
        assert _SESSION_ID_PATTERN.fullmatch("ses") is None  # too short, no chars after prefix
        assert _SESSION_ID_PATTERN.fullmatch("abc123") is None  # missing ses prefix
        assert _SESSION_ID_PATTERN.fullmatch("ses abc") is None  # space

    def test_valid_message_id_passes(self):
        from nexus_mcp.opencode_resources import _MESSAGE_ID_PATTERN

        assert _MESSAGE_ID_PATTERN.fullmatch("msg_abc123")
        assert _MESSAGE_ID_PATTERN.fullmatch("msg-def-456")

    def test_invalid_message_id_fails(self):
        from nexus_mcp.opencode_resources import _MESSAGE_ID_PATTERN

        assert _MESSAGE_ID_PATTERN.fullmatch("../../etc/passwd") is None
        assert _MESSAGE_ID_PATTERN.fullmatch("") is None
        assert _MESSAGE_ID_PATTERN.fullmatch("msg") is None
        assert _MESSAGE_ID_PATTERN.fullmatch("ses_abc") is None  # wrong prefix


class TestSessionTodoResource:
    @respx.mock
    async def test_returns_todo_list(self):
        respx.get("http://test:4096/session/ses_abc/todo").mock(
            return_value=httpx.Response(200, json=[{"text": "Fix auth bug", "completed": False}])
        )
        from nexus_mcp.opencode_resources import get_session_todo

        result = await get_session_todo(session_id="ses_abc")
        assert "Fix auth bug" in result

    async def test_invalid_session_id_raises_resource_error(self):
        from nexus_mcp.opencode_resources import get_session_todo

        with pytest.raises(ResourceError, match="Invalid session_id"):
            await get_session_todo(session_id="../../etc/passwd")


class TestSessionMessagesResource:
    @respx.mock
    async def test_returns_message_list(self):
        respx.get("http://test:4096/session/ses_abc/message").mock(
            return_value=httpx.Response(
                200,
                json=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            )
        )
        from nexus_mcp.opencode_resources import get_session_messages

        result = await get_session_messages(session_id="ses_abc")
        assert "hello" in result
        assert "assistant" in result

    async def test_invalid_session_id_raises_resource_error(self):
        from nexus_mcp.opencode_resources import get_session_messages

        with pytest.raises(ResourceError, match="Invalid session_id"):
            await get_session_messages(session_id="../hack")


class TestSessionChildrenResource:
    @respx.mock
    async def test_returns_children_list(self):
        respx.get("http://test:4096/session/ses_abc/children").mock(
            return_value=httpx.Response(200, json=[{"id": "ses_child1"}])
        )
        from nexus_mcp.opencode_resources import get_session_children

        result = await get_session_children(session_id="ses_abc")
        assert "ses_child1" in result


class TestSessionDiffResource:
    @respx.mock
    async def test_returns_diff(self):
        respx.get("http://test:4096/session/ses_abc/diff").mock(
            return_value=httpx.Response(200, json={"diff": "--- a/file.py\n+++ b/file.py"})
        )
        from nexus_mcp.opencode_resources import get_session_diff

        result = await get_session_diff(session_id="ses_abc")
        assert "file.py" in result


class TestSessionMessageResource:
    @respx.mock
    async def test_returns_single_message(self):
        respx.get("http://test:4096/session/ses_abc/message/msg_123").mock(
            return_value=httpx.Response(
                200, json={"id": "msg_123", "role": "user", "content": "fix bug"}
            )
        )
        from nexus_mcp.opencode_resources import get_session_message

        result = await get_session_message(session_id="ses_abc", message_id="msg_123")
        assert "fix bug" in result

    async def test_invalid_session_id_raises_resource_error(self):
        from nexus_mcp.opencode_resources import get_session_message

        with pytest.raises(ResourceError, match="Invalid session_id"):
            await get_session_message(session_id="../hack", message_id="msg_123")

    async def test_invalid_message_id_raises_resource_error(self):
        from nexus_mcp.opencode_resources import get_session_message

        with pytest.raises(ResourceError, match="Invalid message_id"):
            await get_session_message(session_id="ses_abc", message_id="../hack")
