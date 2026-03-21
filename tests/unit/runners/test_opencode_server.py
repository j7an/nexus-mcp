# tests/unit/runners/test_opencode_server.py
"""Tests for OpenCodeServerRunner.

Mock boundary: httpx.AsyncClient — all HTTP calls mocked.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nexus_mcp.exceptions import RetryableError, SubprocessError
from nexus_mcp.runners.factory import RunnerFactory
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


def _text_part_event(
    session_id: str, text: str, part_id: str = "prt_1", msg_id: str = "msg_1"
) -> str:
    """Build a message.part.updated SSE event for a completed text part."""
    return json.dumps(
        {
            "payload": {
                "type": "message.part.updated",
                "properties": {
                    "part": {
                        "id": part_id,
                        "sessionID": session_id,
                        "messageID": msg_id,
                        "type": "text",
                        "text": text,
                        "time": {"start": 0, "end": 1},
                    }
                },
            }
        }
    )


def _idle_event(session_id: str) -> str:
    """Build a session.idle SSE event (signals completion)."""
    return json.dumps(
        {"payload": {"type": "session.idle", "properties": {"sessionID": session_id}}}
    )


def _message_updated_event(session_id: str, msg_id: str, role: str) -> str:
    """Build a message.updated SSE event (announces a message and its role)."""
    return json.dumps(
        {
            "payload": {
                "type": "message.updated",
                "properties": {
                    "info": {
                        "id": msg_id,
                        "sessionID": session_id,
                        "role": role,
                    }
                },
            }
        }
    )


def _error_event(session_id: str, name: str, message: str, status_code: int) -> str:
    """Build an error SSE event."""
    return json.dumps(
        {
            "payload": {
                "type": "error",
                "properties": {
                    "sessionID": session_id,
                    "error": {
                        "name": name,
                        "data": {"message": message, "statusCode": status_code},
                    },
                },
            }
        }
    )


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

        # Mock SSE stream: assistant message announcement, text part, then idle
        sse_text = _sse_lines(
            _message_updated_event("ses_test", "msg_1", "assistant"),
            _text_part_event("ses_test", "Hello world"),
            _idle_event("ses_test"),
        )
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

        sse_text = _sse_lines(_text_part_event("ses_test", "ok"), _idle_event("ses_test"))
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

        sse_text = _sse_lines(_text_part_event("ses_test", "ok"), _idle_event("ses_test"))
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
        assert body["model"] == {"modelID": "claude-3-opus", "providerID": "anthropic"}

    async def test_sse_stream_opened_before_prompt_async(self):
        """SSE must connect before prompt fires to avoid missing step_finish."""
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_test")

        call_order: list[str] = []

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204
        prompt_resp.raise_for_status = MagicMock()

        async def recording_post(*args, **kwargs):
            call_order.append("post")
            return prompt_resp

        runner._client.post = AsyncMock(side_effect=recording_post)

        sse_text = _sse_lines(_text_part_event("ses_test", "ok"), _idle_event("ses_test"))
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def recording_stream(method, url, **kwargs):
            call_order.append("stream")
            yield sse_resp

        runner._client.stream = recording_stream

        request = make_prompt_request(cli="opencode_server", prompt="hi")
        await runner._execute(request)

        assert call_order == ["stream", "post"], (
            f"SSE stream must open before prompt_async, got order: {call_order}"
        )


class TestErrorHandling:
    """Test HTTP and SSE error classification."""

    async def test_connect_error_is_retryable(self):
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_test")
        runner._client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(RetryableError, match="Cannot connect"):
            request = make_prompt_request(cli="opencode_server")
            await runner._execute(request)

    async def test_timeout_error_is_retryable(self):
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_test")
        runner._client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with pytest.raises(RetryableError, match="timed out"):
            request = make_prompt_request(cli="opencode_server")
            await runner._execute(request)

    async def test_http_429_raises_retryable(self):
        runner = make_server_runner()
        resp = AsyncMock(spec=httpx.Response)
        resp.status_code = 429
        resp.headers = {"Retry-After": "5"}

        with pytest.raises(RetryableError) as exc_info:
            runner._check_http_error(resp)
        assert exc_info.value.retry_after == 5.0

    async def test_http_503_is_retryable(self):
        runner = make_server_runner()
        resp = AsyncMock(spec=httpx.Response)
        resp.status_code = 503
        resp.headers = {}

        with pytest.raises(RetryableError):
            runner._check_http_error(resp)

    async def test_http_401_is_not_retryable(self):
        runner = make_server_runner()
        resp = AsyncMock(spec=httpx.Response)
        resp.status_code = 401
        resp.headers = {}
        resp.text = "Unauthorized"

        with pytest.raises(SubprocessError, match="401"):
            runner._check_http_error(resp)

    async def test_http_403_is_not_retryable(self):
        runner = make_server_runner()
        resp = AsyncMock(spec=httpx.Response)
        resp.status_code = 403
        resp.headers = {}
        resp.text = "Forbidden"

        with pytest.raises(SubprocessError, match="403"):
            runner._check_http_error(resp)


class TestAuthConfiguration:
    """Test HTTP client auth setup."""

    def test_no_auth_when_password_unset(self):
        runner = make_server_runner()
        assert runner._client.auth is None

    @patch.dict("os.environ", {"NEXUS_OPENCODE_SERVER_PASSWORD": "secret"})
    def test_basic_auth_when_password_set(self):
        runner = make_server_runner()
        assert runner._client.auth is not None


class TestSSEErrorEvents:
    """Test SSE error event classification."""

    async def test_sse_error_event_raises_subprocess_error(self):
        """An SSE error event with non-retryable code raises SubprocessError."""
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_test")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204
        prompt_resp.raise_for_status = MagicMock()

        sse_text = _sse_lines(
            _error_event("ses_test", "AuthenticationError", "Invalid API key", 401)
        )
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        request = make_prompt_request(cli="opencode_server")
        with pytest.raises(SubprocessError, match="AuthenticationError"):
            await runner._execute(request)

    async def test_sse_error_event_retryable_code(self):
        """An SSE error event with 429 raises RetryableError."""
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_test")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204
        prompt_resp.raise_for_status = MagicMock()

        sse_text = _sse_lines(_error_event("ses_test", "RateLimitError", "Too many requests", 429))
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        request = make_prompt_request(cli="opencode_server")
        with pytest.raises(RetryableError, match="RateLimitError"):
            await runner._execute(request)


class TestSSEEventParsing:
    """Test _consume_sse parses the real OpenCode event schema."""

    async def test_text_from_message_part_updated(self):
        """Text is extracted from message.part.updated events with type=text."""
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_abc")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204

        sse_text = _sse_lines(
            _message_updated_event("ses_abc", "msg_1", "assistant"),
            _text_part_event("ses_abc", "I am kimi", part_id="prt_1"),
            _idle_event("ses_abc"),
        )
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        request = make_prompt_request(cli="opencode_server")
        response = await runner._execute(request)
        assert response.output == "I am kimi"

    async def test_events_from_other_session_ignored(self):
        """Events whose sessionID doesn't match are skipped; only own idle triggers completion."""
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_mine")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204

        sse_text = _sse_lines(
            _message_updated_event("ses_mine", "msg_1", "assistant"),
            _text_part_event("ses_other", "wrong"),  # different session — ignore
            _text_part_event("ses_mine", "correct"),  # our session — keep
            _idle_event("ses_other"),  # other session idle — ignore
            _idle_event("ses_mine"),  # our session idle — stop
        )
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        request = make_prompt_request(cli="opencode_server")
        response = await runner._execute(request)
        assert response.output == "correct"

    async def test_user_message_part_excluded_from_output(self):
        """Text parts belonging to the user message must not appear in output."""
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_x")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204

        sse_text = _sse_lines(
            _message_updated_event("ses_x", "msg_user", "user"),
            _text_part_event("ses_x", "what model are you", part_id="prt_u", msg_id="msg_user"),
            _message_updated_event("ses_x", "msg_asst", "assistant"),
            _text_part_event("ses_x", "I am kimi", part_id="prt_a", msg_id="msg_asst"),
            _idle_event("ses_x"),
        )
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        response = await runner._execute(make_prompt_request(cli="opencode_server"))
        assert response.output == "I am kimi"
        assert "what model are you" not in response.output

    async def test_output_empty_when_no_assistant_message_announced(self):
        """If no message.updated/assistant event arrives, no text is collected."""
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_x")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204

        # Text part arrives but no message.updated/assistant to associate it with
        sse_text = _sse_lines(
            _text_part_event("ses_x", "orphaned text", part_id="prt_1", msg_id="msg_unknown"),
            _idle_event("ses_x"),
        )
        sse_resp = _mock_sse_response(sse_text)

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream

        response = await runner._execute(make_prompt_request(cli="opencode_server"))
        assert response.output == ""


class TestParseModel:
    """Test _parse_model splits 'provider/model' into the API object format."""

    def test_provider_slash_model_splits_correctly(self):
        runner = make_server_runner()
        result = runner._parse_model("ollama-cloud/kimi-k2.5")
        assert result == {"modelID": "kimi-k2.5", "providerID": "ollama-cloud"}

    def test_model_without_slash_uses_empty_provider(self):
        runner = make_server_runner()
        result = runner._parse_model("kimi-k2.5")
        assert result == {"modelID": "kimi-k2.5", "providerID": ""}

    def test_multiple_slashes_splits_on_first(self):
        runner = make_server_runner()
        result = runner._parse_model("provider/model/variant")
        assert result == {"modelID": "model/variant", "providerID": "provider"}


class TestFactoryRegistration:
    """Test that opencode_server is registered in RunnerFactory."""

    def test_opencode_server_in_list_clis(self):
        assert "opencode_server" in RunnerFactory.list_clis()

    def test_create_returns_opencode_server_runner(self):
        runner = RunnerFactory.create("opencode_server")
        assert isinstance(runner, OpenCodeServerRunner)
        assert runner.AGENT_NAME == "opencode_server"

    def test_get_runner_class(self):
        cls = RunnerFactory.get_runner_class("opencode_server")
        assert cls is OpenCodeServerRunner


class TestCleanupAbort:
    """Test that _consume_sse aborts the session on cancellation/timeout."""

    async def test_abort_called_on_cancellation(self):
        """CancelledError during SSE triggers abort and re-raises."""
        runner = make_server_runner()
        runner._resolve_session = AsyncMock(return_value="ses_cancel")

        prompt_resp = AsyncMock(spec=httpx.Response)
        prompt_resp.status_code = 204

        # SSE stream that raises CancelledError during iteration
        sse_resp = AsyncMock(spec=httpx.Response)
        sse_resp.status_code = 200

        async def cancelling_aiter_lines():
            yield "data: {}\n"
            raise asyncio.CancelledError

        sse_resp.aiter_lines = cancelling_aiter_lines

        @asynccontextmanager
        async def mock_stream(method, url, **kwargs):
            yield sse_resp

        abort_mock = AsyncMock()
        runner._client.post = AsyncMock(return_value=prompt_resp)
        runner._client.stream = mock_stream
        runner._abort_session = abort_mock

        request = make_prompt_request(cli="opencode_server")
        with pytest.raises(asyncio.CancelledError):
            await runner._execute(request)

        abort_mock.assert_awaited_once_with("ses_cancel")
