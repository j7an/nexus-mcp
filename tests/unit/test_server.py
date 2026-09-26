import asyncio
import json
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp import backends, server
from nexus_mcp.types import BackendInfo, PromptRequest, PromptResult


class FakeBackend:
    def __init__(
        self,
        *,
        delay: float = 0,
        error: BaseException | None = None,
        announce: str | None = None,
    ) -> None:
        self.requests: list[PromptRequest] = []
        self.delay = delay
        self.error = error
        self.announce = announce

    async def run(self, req: PromptRequest, on_session) -> PromptResult:
        self.requests.append(req)
        if self.announce is not None:
            on_session(self.announce)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return PromptResult(backend="claude", session_id="sid-1", output="ok")

    async def info(self) -> BackendInfo:
        return BackendInfo(name="claude", installed=True)


@pytest.fixture
def fake(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(backends, "installed", lambda name: True)
    monkeypatch.setattr(backends, "get", lambda name: backend)
    return backend


async def test_prompt_dispatches_validated_request(fake, tmp_path):
    result = await server.run_prompt(backend="claude", prompt="hi", cwd=str(tmp_path))
    assert result == PromptResult(backend="claude", session_id="sid-1", output="ok")
    [req] = fake.requests
    assert req == PromptRequest(prompt="hi", cwd=tmp_path.resolve(), profile="read_only")


async def test_prompt_passes_session_fork_model_profile(fake, tmp_path):
    await server.run_prompt(
        backend="claude",
        prompt="hi",
        cwd=str(tmp_path),
        profile="workspace_write",
        session_id="abc-123",
        fork=True,
        model="haiku",
    )
    [req] = fake.requests
    assert (req.profile, req.session_id, req.fork, req.model) == (
        "workspace_write",
        "abc-123",
        True,
        "haiku",
    )


@pytest.mark.parametrize("make_cwd", ["relative", "file", "missing", "nul", "too_long"])
async def test_rejects_bad_cwd(fake, tmp_path, make_cwd):
    (tmp_path / "f.txt").write_text("x")
    cwd = {
        "relative": "some/dir",
        "file": str(tmp_path / "f.txt"),
        "missing": str(tmp_path / "nope"),
        "nul": str(tmp_path) + "\x00x",
        "too_long": str(tmp_path / ("a" * 300)),
    }[make_cwd]
    with pytest.raises(ToolError, match="cwd"):
        await server.run_prompt(backend="claude", prompt="hi", cwd=cwd)
    assert fake.requests == []


@pytest.mark.parametrize("session_id", ["-rf", "a b", "x" * 300, "", "id;rm"])
async def test_rejects_bad_session_id(fake, tmp_path, session_id):
    with pytest.raises(ToolError, match="session_id"):
        await server.run_prompt(
            backend="claude", prompt="hi", cwd=str(tmp_path), session_id=session_id
        )
    assert fake.requests == []


async def test_fork_requires_session_id(fake, tmp_path):
    with pytest.raises(ToolError, match="fork"):
        await server.run_prompt(backend="claude", prompt="hi", cwd=str(tmp_path), fork=True)


async def test_uninstalled_backend_gets_install_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "installed", lambda name: False)
    with pytest.raises(ToolError, match=r"nexus-mcp\[claude\]"):
        await server.run_prompt(backend="claude", prompt="hi", cwd=str(tmp_path))


async def test_timeout_mentions_session_when_continuing(monkeypatch, tmp_path):
    slow = FakeBackend(delay=5)
    monkeypatch.setattr(backends, "installed", lambda name: True)
    monkeypatch.setattr(backends, "get", lambda name: slow)
    with pytest.raises(ToolError, match=r"timed out after 1s.*session_id=abc-123"):
        await server.run_prompt(
            backend="claude", prompt="hi", cwd=str(tmp_path), session_id="abc-123", timeout=1
        )


async def test_timeout_without_session_has_no_resume_hint(monkeypatch, tmp_path):
    slow = FakeBackend(delay=5)
    monkeypatch.setattr(backends, "installed", lambda name: True)
    monkeypatch.setattr(backends, "get", lambda name: slow)
    with pytest.raises(ToolError) as info:
        await server.run_prompt(backend="claude", prompt="hi", cwd=str(tmp_path), timeout=1)
    assert "timed out after 1s" in str(info.value)
    assert "session_id" not in str(info.value)


@pytest.mark.parametrize(
    ("session_id", "fork"), [(None, False), ("abc-123", True)], ids=["new", "fork"]
)
async def test_timeout_reports_session_acquired_by_backend(monkeypatch, tmp_path, session_id, fork):
    slow = FakeBackend(delay=5, announce="sid-acquired")
    monkeypatch.setattr(backends, "installed", lambda name: True)
    monkeypatch.setattr(backends, "get", lambda name: slow)
    with pytest.raises(ToolError) as info:
        await server.run_prompt(
            backend="claude",
            prompt="hi",
            cwd=str(tmp_path),
            session_id=session_id,
            fork=fork,
            timeout=1,
        )
    assert "session_id=sid-acquired" in str(info.value)
    assert "abc-123" not in str(info.value)


async def test_inner_timeout_error_is_not_reported_as_deadline(monkeypatch, tmp_path):
    failing = FakeBackend(error=TimeoutError("inner"))
    monkeypatch.setattr(backends, "installed", lambda name: True)
    monkeypatch.setattr(backends, "get", lambda name: failing)
    with pytest.raises(TimeoutError, match="inner"):
        await server.run_prompt(backend="claude", prompt="hi", cwd=str(tmp_path))


async def test_backends_resource_lists_uninstalled_with_hint(monkeypatch):
    monkeypatch.setattr(backends, "installed", lambda name: False)
    payload = json.loads(await server.backends_resource())
    assert payload == [
        {
            "name": "claude",
            "installed": False,
            "models": None,
            "hint": backends.install_hint("claude"),
        }
    ]


async def test_backends_resource_uses_backend_info(fake):
    payload = json.loads(await server.backends_resource())
    assert payload == [{"name": "claude", "installed": True, "models": None, "hint": None}]


def test_instructions_list_installed_backends(monkeypatch):
    monkeypatch.setattr(backends, "installed", lambda name: True)
    text = server.build_instructions()
    assert "Installed backends: claude." in text
    assert "`cwd` is required" in text
    assert server.mcp.instructions is not None
    assert "`cwd` is required" in server.mcp.instructions
    monkeypatch.setattr(backends, "installed", lambda name: False)
    assert "Installed backends: none" in server.build_instructions()


def test_installed_reflects_sdk_importability(monkeypatch):
    monkeypatch.setattr(backends, "find_spec", lambda name: None)
    assert backends.installed("claude") is False
    monkeypatch.setattr(backends, "find_spec", lambda name: SimpleNamespace())
    assert backends.installed("claude") is True


async def test_timeout_omits_invalid_backend_session_id(monkeypatch, tmp_path):
    slow = FakeBackend(delay=5, announce="sid-acquired\nprovider detail")
    monkeypatch.setattr(backends, "installed", lambda name: True)
    monkeypatch.setattr(backends, "get", lambda name: slow)
    with pytest.raises(ToolError) as info:
        await server.run_prompt(backend="claude", prompt="hi", cwd=str(tmp_path), timeout=1)
    assert "timed out after 1s" in str(info.value)
    assert "session_id" not in str(info.value)
    assert "provider detail" not in str(info.value)


@pytest.mark.parametrize("operation", ["is_dir", "resolve"])
@pytest.mark.parametrize("error", [PermissionError, ValueError])
async def test_cwd_filesystem_errors_are_sanitized(fake, tmp_path, monkeypatch, operation, error):
    def fail(_path):
        raise error("PRIVATE filesystem detail")

    with monkeypatch.context() as patch:
        patch.setattr(type(tmp_path), operation, fail)
        with pytest.raises(ToolError, match="^cwd must be an existing directory$"):
            await server.run_prompt(backend="claude", prompt="hi", cwd=str(tmp_path))
    assert fake.requests == []
