"""Tests for compound tools with sampling fallback."""

from unittest.mock import AsyncMock

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


class TestOpenCodeInvestigate:
    @respx.mock
    async def test_chains_search_and_file_read(self):
        respx.get("http://test:4096/find").mock(
            return_value=httpx.Response(
                200, json=[{"path": "src/auth.py"}, {"path": "src/login.py"}]
            )
        )
        respx.get("http://test:4096/file/content").mock(
            return_value=httpx.Response(200, json={"content": "def authenticate(): pass"})
        )
        from nexus_mcp.compound_tools import opencode_investigate

        result = await opencode_investigate(query="auth", max_files=2)
        assert "auth.py" in result
        assert "authenticate" in result

    @respx.mock
    async def test_respects_max_files_limit(self):
        respx.get("http://test:4096/find").mock(
            return_value=httpx.Response(
                200,
                json=[{"path": f"file{i}.py"} for i in range(10)],
            )
        )
        respx.get("http://test:4096/file/content").mock(
            return_value=httpx.Response(200, json={"content": "code"})
        )
        from nexus_mcp.compound_tools import opencode_investigate

        await opencode_investigate(query="test", max_files=3)
        assert respx.calls.call_count == 4  # 1 search + 3 file reads

    @respx.mock
    async def test_sampling_success_returns_analyzed_result(self):
        respx.get("http://test:4096/find").mock(
            return_value=httpx.Response(200, json=[{"path": "src/auth.py"}])
        )
        respx.get("http://test:4096/file/content").mock(
            return_value=httpx.Response(200, json={"content": "code"})
        )
        mock_ctx = AsyncMock()
        mock_sample_result = AsyncMock()
        mock_sample_result.text = "Analysis: auth module handles authentication"
        mock_ctx.sample.return_value = mock_sample_result
        from nexus_mcp.compound_tools import opencode_investigate

        result = await opencode_investigate(query="auth", ctx=mock_ctx)
        assert "Analysis:" in result

    @respx.mock
    async def test_sampling_failure_returns_raw_data(self):
        respx.get("http://test:4096/find").mock(
            return_value=httpx.Response(200, json=[{"path": "src/auth.py"}])
        )
        respx.get("http://test:4096/file/content").mock(
            return_value=httpx.Response(200, json={"content": "code"})
        )
        mock_ctx = AsyncMock()
        mock_ctx.sample.side_effect = Exception("sampling not supported")
        from nexus_mcp.compound_tools import opencode_investigate

        result = await opencode_investigate(query="auth", ctx=mock_ctx)
        assert "auth.py" in result


class TestOpenCodeSessionReview:
    @respx.mock
    async def test_chains_session_messages_diff(self):
        respx.get("http://test:4096/session/ses_1").mock(
            return_value=httpx.Response(200, json={"id": "ses_1", "status": "completed"})
        )
        respx.get("http://test:4096/session/ses_1/message").mock(
            return_value=httpx.Response(200, json=[{"role": "user", "content": "fix the bug"}])
        )
        respx.get("http://test:4096/session/ses_1/diff").mock(
            return_value=httpx.Response(200, json={"diff": "--- a/file.py\n+++ b/file.py"})
        )
        respx.get("http://test:4096/session/ses_1/todo").mock(
            return_value=httpx.Response(200, json=[])
        )
        from nexus_mcp.compound_tools import opencode_session_review

        result = await opencode_session_review(session_id="ses_1")
        assert "ses_1" in result
        assert "fix the bug" in result

    @respx.mock
    async def test_sampling_fallback(self):
        respx.get("http://test:4096/session/ses_1").mock(
            return_value=httpx.Response(200, json={"id": "ses_1", "status": "completed"})
        )
        respx.get("http://test:4096/session/ses_1/message").mock(
            return_value=httpx.Response(200, json=[])
        )
        respx.get("http://test:4096/session/ses_1/diff").mock(
            return_value=httpx.Response(200, json={"diff": ""})
        )
        respx.get("http://test:4096/session/ses_1/todo").mock(
            return_value=httpx.Response(200, json=[])
        )
        mock_ctx = AsyncMock()
        mock_ctx.sample.side_effect = Exception("not supported")
        from nexus_mcp.compound_tools import opencode_session_review

        result = await opencode_session_review(session_id="ses_1", ctx=mock_ctx)
        assert "ses_1" in result

    @respx.mock
    async def test_includes_todo_data(self):
        respx.get("http://test:4096/session/ses_abc").mock(
            return_value=httpx.Response(200, json={"id": "ses_abc", "status": "completed"})
        )
        respx.get("http://test:4096/session/ses_abc/message").mock(
            return_value=httpx.Response(200, json=[])
        )
        respx.get("http://test:4096/session/ses_abc/diff").mock(
            return_value=httpx.Response(200, json={"diff": ""})
        )
        respx.get("http://test:4096/session/ses_abc/todo").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"text": "Fix auth bug", "completed": True},
                    {"text": "Add tests", "completed": False},
                ],
            )
        )
        from nexus_mcp.compound_tools import opencode_session_review

        result = await opencode_session_review(session_id="ses_abc")
        assert "Fix auth bug" in result
        assert "Add tests" in result
        assert "✓" in result  # completed marker
        assert "○" in result  # incomplete marker

    @respx.mock
    async def test_empty_todos_omitted(self):
        respx.get("http://test:4096/session/ses_abc").mock(
            return_value=httpx.Response(200, json={"id": "ses_abc", "status": "completed"})
        )
        respx.get("http://test:4096/session/ses_abc/message").mock(
            return_value=httpx.Response(200, json=[])
        )
        respx.get("http://test:4096/session/ses_abc/diff").mock(
            return_value=httpx.Response(200, json={"diff": ""})
        )
        respx.get("http://test:4096/session/ses_abc/todo").mock(
            return_value=httpx.Response(200, json=[])
        )
        from nexus_mcp.compound_tools import opencode_session_review

        result = await opencode_session_review(session_id="ses_abc")
        assert "Todos" not in result
