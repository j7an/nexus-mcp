"""In-process MCP clients against the real server with a fake backend."""

import pytest
from fastmcp import Client

from nexus_mcp import backends
from nexus_mcp.server import mcp
from nexus_mcp.types import BackendInfo, PromptRequest, PromptResult


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[PromptRequest] = []

    async def run(self, req: PromptRequest, on_session) -> PromptResult:
        self.requests.append(req)
        return PromptResult(
            backend="claude",
            session_id="sid-1",
            output=f"echo: {req.prompt}",
            usage={"n": 1},
        )

    async def info(self) -> BackendInfo:
        return BackendInfo(name="claude", installed=True)


@pytest.fixture
def fake_backend(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr(backends, "installed", lambda name: True)
    monkeypatch.setattr(backends, "get", lambda name: fake)
    return fake


@pytest.fixture(params=["auto", "legacy"])
async def client(request, fake_backend):
    """Exercise both modern and legacy handshake protocol eras."""
    async with Client(mcp, mode=request.param) as connected:
        yield connected
