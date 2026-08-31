"""Contracts for deterministic durable-job MCP tool registration."""

import getpass
import json
import platform
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.core import UnsupportedCapabilityError, WorkspaceSelector
from nexus_mcp.jobs import EventPollingPolicy, WorkerPolicy
from nexus_mcp.mcp.server import mcp
from nexus_mcp.runners.factory import RunnerFactory

REQUIRED_AGENT_TOOLS = {
    "agent_start",
    "agent_continue",
    "agent_fork",
    "agent_review",
    "agent_diagnose",
    "agent_status",
    "agent_result",
    "agent_cancel",
    "agent_respond",
    "agent_list",
    "agent_backends",
}


def _fast_tuning():
    from nexus_mcp.mcp.runtime import RuntimeTuning

    return RuntimeTuning(
        worker_policy=WorkerPolicy(
            lease_seconds=1.0,
            heartbeat_seconds=0.2,
            idle_poll_seconds=0.001,
            reconciliation_timeout_seconds=1.0,
        ),
        event_polling_policy=EventPollingPolicy(
            minimum_seconds=0.001,
            maximum_seconds=0.005,
        ),
        retry_delay=lambda _attempt, _retry_after, _policy: 0.0,
    )


def test_local_principal_uses_posix_user_id(monkeypatch):
    """POSIX callers are identified without accepting a tool-supplied principal."""
    from nexus_mcp.mcp import access

    monkeypatch.setattr(access.os, "getuid", lambda: 712)

    assert access.local_principal_id() == "local:712"


def test_local_access_context_authorizes_local_workspaces(monkeypatch):
    """The adapter constructs one local-trust access context internally."""
    from nexus_mcp.mcp import access

    monkeypatch.setattr(access, "local_principal_id", lambda: "local:712")

    assert access.local_access_context().model_dump(mode="json") == {
        "principal_id": "local:712",
        "authentication_kind": "local",
        "roles": [],
        "authorized_workspace_ids": [],
        "authorize_local_workspaces": True,
    }


def test_local_principal_has_deterministic_windows_fallback(monkeypatch):
    """Platforms without getuid bind identity to normalized machine and account."""
    from nexus_mcp.mcp import access

    monkeypatch.delattr(access.os, "getuid", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: " Example.User ")
    monkeypatch.setattr(platform, "node", lambda: " WorkStation ")

    assert access.local_principal_id() == "local-windows:workstation:example.user"


async def test_runtime_provider_reuses_installed_runtime():
    """A lifespan-owned runtime is borrowed without opening or owning another one."""
    from nexus_mcp.mcp.runtime import RuntimeProvider

    provider = RuntimeProvider()
    installed = object()

    async with (
        provider.install(installed),
        provider.borrow() as outer,
        provider.borrow() as inner,
    ):
        assert outer is installed
        assert inner is installed


async def test_runtime_provider_reuses_and_closes_one_temporary_runtime(monkeypatch):
    """Nested direct/Docket borrows share exactly one provider-owned runtime."""
    from nexus_mcp.mcp import runtime

    provider = runtime.RuntimeProvider()
    temporary = object()
    lifecycle: list[tuple[str, object]] = []

    @asynccontextmanager
    async def fake_open(tuning):
        lifecycle.append(("open", tuning))
        try:
            yield temporary
        finally:
            lifecycle.append(("close", tuning))

    monkeypatch.setattr(runtime.MCPRuntime, "open", staticmethod(fake_open))

    async with provider.borrow() as outer, provider.borrow() as inner:
        assert outer is temporary
        assert inner is temporary
        assert [event for event, _tuning in lifecycle] == ["open"]

    assert [event for event, _tuning in lifecycle] == ["open", "close"]


async def test_runtime_provider_temporary_runtime_closes_after_tool_error(monkeypatch):
    """A failed direct tool call cannot leak its fallback runtime."""
    from nexus_mcp.mcp import runtime

    provider = runtime.RuntimeProvider()
    lifecycle: list[str] = []

    @asynccontextmanager
    async def fake_open(_tuning):
        lifecycle.append("open")
        try:
            yield object()
        finally:
            lifecycle.append("close")

    monkeypatch.setattr(runtime.MCPRuntime, "open", staticmethod(fake_open))

    with pytest.raises(RuntimeError, match="tool failed"):
        async with provider.borrow():
            raise RuntimeError("tool failed")

    assert lifecycle == ["open", "close"]


async def test_override_tuning_drives_fallback_and_restores(monkeypatch):
    """One scoped tuning override reaches fallback opens and restores afterward."""
    from nexus_mcp.mcp import runtime

    provider = runtime.RuntimeProvider()
    original = provider.tuning
    override = _fast_tuning()
    observed = []

    @asynccontextmanager
    async def fake_open(tuning):
        observed.append(tuning)
        yield object()

    monkeypatch.setattr(runtime.MCPRuntime, "open", staticmethod(fake_open))

    with provider.override_tuning(override):
        async with provider.borrow():
            pass
        assert provider.tuning is override

    assert observed == [override]
    assert provider.tuning is original


async def test_override_tuning_rejects_an_installed_runtime():
    """Timing dependencies cannot be replaced after a runtime has started."""
    from nexus_mcp.mcp.runtime import RuntimeProvider

    provider = RuntimeProvider()

    async with provider.install(object()):
        with pytest.raises(RuntimeError, match="runtime is installed"):
            with provider.override_tuning(_fast_tuning()):
                pass


async def test_mcp_runtime_injects_tuning_and_closes_in_reverse(monkeypatch):
    """Runtime dependencies start in order and always unwind worker/backend/store."""
    from nexus_mcp.mcp import runtime

    tuning = _fast_tuning()
    lifecycle: list[str] = []
    fake_legacy_backends = (object(),)

    class FakeStore:
        def __init__(self):
            lifecycle.append("store.construct")

        async def open(self):
            lifecycle.append("store.open")

        async def close(self):
            lifecycle.append("store.close")

    class FakeBackendManager:
        def __init__(self, backends):
            assert backends is fake_legacy_backends
            lifecycle.append("backends.construct")

        async def close(self):
            lifecycle.append("backends.close")

    class FakeNotifier:
        def __init__(self):
            lifecycle.append("notifier.construct")

    class FakeResolver:
        def __init__(self):
            lifecycle.append("resolver.construct")

    class FakeService:
        def __init__(self, *, store, backend_manager, config_resolver, notifier):
            assert isinstance(store, FakeStore)
            assert isinstance(backend_manager, FakeBackendManager)
            assert isinstance(config_resolver, FakeResolver)
            assert isinstance(notifier, FakeNotifier)
            lifecycle.append("service.construct")

    class FakeWorkers:
        def __init__(self, *, store, backends, notifier, policy, retry_delay):
            assert isinstance(store, FakeStore)
            assert isinstance(backends, FakeBackendManager)
            assert isinstance(notifier, FakeNotifier)
            assert policy is tuning.worker_policy
            assert retry_delay is tuning.retry_delay
            lifecycle.append("workers.construct")

        async def start(self):
            lifecycle.append("workers.start")

        async def stop(self):
            lifecycle.append("workers.stop")

    def fake_legacy_factory():
        lifecycle.append("legacy_backends.construct")
        return fake_legacy_backends

    monkeypatch.setattr(runtime, "_MCPJobStore", FakeStore)
    monkeypatch.setattr(runtime, "BackendManager", FakeBackendManager)
    monkeypatch.setattr(runtime, "EventNotifier", FakeNotifier)
    monkeypatch.setattr(runtime, "NexusConfigResolver", FakeResolver)
    monkeypatch.setattr(runtime, "AgentJobService", FakeService)
    monkeypatch.setattr(runtime, "WorkerPool", FakeWorkers)
    monkeypatch.setattr(runtime, "legacy_backends", fake_legacy_factory)

    async with runtime.MCPRuntime.open(tuning) as opened:
        assert opened.event_polling_policy is tuning.event_polling_policy
        assert lifecycle == [
            "store.construct",
            "store.open",
            "legacy_backends.construct",
            "backends.construct",
            "notifier.construct",
            "resolver.construct",
            "service.construct",
            "workers.construct",
            "workers.start",
        ]

    assert lifecycle[-3:] == ["workers.stop", "backends.close", "store.close"]


async def test_mcp_runtime_runs_real_workers_and_closes_store(fake_runner_registry):
    """The composed runtime owns live workers and its SQLite connection."""
    from nexus_mcp.mcp.runtime import MCPRuntime

    del fake_runner_registry
    tuning = _fast_tuning()
    async with MCPRuntime.open(tuning) as opened:
        assert opened.workers.running is True
        store = opened.store

    assert opened.workers.running is False
    with pytest.raises(RuntimeError, match="closed"):
        await store.get_job("missing-job")


async def test_agent_tools_registered_when_all_backends_unavailable(monkeypatch):
    """Backend discovery cannot remove the stable durable-job tool surface."""
    monkeypatch.setattr(RunnerFactory, "_REGISTRY", {})

    names = {tool.name for tool in await mcp.list_tools()}

    assert names >= REQUIRED_AGENT_TOOLS


async def test_agent_tool_schemas_require_workspace_and_never_accept_principal():
    """Every durable tool has an explicit workspace and adapter-owned identity."""
    tools = {
        tool.name: tool for tool in await mcp.list_tools() if tool.name in REQUIRED_AGENT_TOOLS
    }

    assert set(tools) == REQUIRED_AGENT_TOOLS
    for name, tool in tools.items():
        properties = tool.parameters["properties"]
        assert "workspace" in tool.parameters["required"], name
        assert "principal_id" not in properties, name
        assert tool.output_schema is not None, name


async def test_agent_tool_annotations_match_observation_and_mutation():
    """Discovery metadata distinguishes read-only polling from execution/control."""
    tools = {
        tool.name: tool for tool in await mcp.list_tools() if tool.name in REQUIRED_AGENT_TOOLS
    }

    for name in {"agent_status", "agent_result", "agent_list", "agent_backends"}:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is True
        assert annotations.destructiveHint is False
        assert annotations.idempotentHint is True
    for name in {"agent_start", "agent_continue", "agent_fork", "agent_review", "agent_diagnose"}:
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.readOnlyHint is False
        assert annotations.destructiveHint is True
        assert annotations.idempotentHint is False


async def test_unsupported_capability_becomes_structured_tool_error(monkeypatch):
    """A stable capability failure remains machine-readable at the MCP boundary."""
    from nexus_mcp.mcp import job_tools

    service = SimpleNamespace(
        fork_session=AsyncMock(side_effect=UnsupportedCapabilityError("fake", "fork"))
    )

    @asynccontextmanager
    async def borrow():
        yield SimpleNamespace(service=service)

    monkeypatch.setattr(job_tools.runtime_provider, "borrow", borrow)

    with pytest.raises(ToolError) as error:
        await job_tools.agent_fork(
            workspace=WorkspaceSelector(path="/tmp"),
            session_id="session-test",
        )

    assert json.loads(str(error.value)) == {
        "code": "unsupported_capability",
        "message": "Backend fake does not support fork",
    }
