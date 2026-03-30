# tests/integration/test_session_integration.py
"""Integration tests for session endpoints against a real OpenCode server.

Requires: `opencode serve` running (or NEXUS_OPENCODE_SERVER_PASSWORD set)
Run: `uv run pytest -m integration tests/integration/test_session_integration.py -v`

All tests are non-destructive:
- Every created session is deleted in fixture teardown
- No file-writing commands (only /help, read-only prompts)
- Share/unshare cleaned up in teardown
"""

import contextlib
import os

import pytest

from nexus_mcp.http_client import OpenCodeHTTPClient, reset_http_client


@pytest.fixture(scope="module")
def opencode_server_available():
    """Skip all tests if OpenCode server is not configured."""
    password = os.environ.get("NEXUS_OPENCODE_SERVER_PASSWORD")
    if not password:
        pytest.skip("NEXUS_OPENCODE_SERVER_PASSWORD not set — skipping server integration tests")


@pytest.fixture(autouse=True)
def _reset(opencode_server_available):  # noqa: ARG001
    reset_http_client()
    yield
    reset_http_client()


@pytest.fixture
async def client():
    """OpenCodeHTTPClient connected to the real server."""
    c = OpenCodeHTTPClient()
    healthy = await c.health_check()
    if not healthy:
        pytest.skip("OpenCode server not reachable")
    yield c
    await c.close()


@pytest.fixture
async def temp_session(client):
    """Create a temporary session, delete on teardown."""
    session_id = await client.create_session()
    yield session_id
    with contextlib.suppress(Exception):
        await client.delete_session(session_id)


@pytest.mark.integration
class TestSessionListing:
    async def test_list_sessions(self, client):
        data = await client.get("/session")
        assert isinstance(data, list)

    async def test_session_status(self, client):
        data = await client.get("/session/status")
        assert isinstance(data, dict)


@pytest.mark.integration
class TestSessionLifecycle:
    async def test_abort_session(self, client, temp_session):
        result = await client.post(f"/session/{temp_session}/abort")
        assert result is not None

    async def test_share_and_unshare_session(self, client, temp_session):
        # Share
        shared = await client.post(f"/session/{temp_session}/share")
        assert shared is not None
        # Unshare (cleanup)
        await client.delete(f"/session/{temp_session}/share")

    async def test_session_todo(self, client, temp_session):
        data = await client.get(f"/session/{temp_session}/todo")
        assert isinstance(data, list)

    async def test_session_diff(self, client, temp_session):
        data = await client.get(f"/session/{temp_session}/diff")
        assert isinstance(data, (dict, list))

    async def test_session_messages(self, client, temp_session):
        data = await client.get(f"/session/{temp_session}/message")
        assert isinstance(data, list)

    async def test_session_children(self, client, temp_session):
        data = await client.get(f"/session/{temp_session}/children")
        assert isinstance(data, list)

    async def test_session_command_help(self, client, temp_session):
        result = await client.post(
            f"/session/{temp_session}/command",
            json={"command": "/help", "args": []},
        )
        assert result is not None


@pytest.mark.integration
class TestSessionFork:
    async def test_fork_and_list_children(self, client, temp_session):
        forked_id = await client.fork_session(temp_session)
        try:
            children = await client.get(f"/session/{temp_session}/children")
            assert isinstance(children, list)
        finally:
            with contextlib.suppress(Exception):
                await client.delete_session(forked_id)


@pytest.mark.integration
class TestPermissionQuestionListing:
    async def test_list_permissions(self, client):
        data = await client.get("/permission")
        assert isinstance(data, list)

    async def test_list_questions(self, client):
        data = await client.get("/question")
        assert isinstance(data, list)
