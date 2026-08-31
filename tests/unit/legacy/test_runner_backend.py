"""Behavioral contracts for the temporary legacy runner backend."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nexus_mcp.backends import BackendFailure
from nexus_mcp.backends.manager import BackendManager
from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.core import (
    BackendEvent,
    JobAttempt,
    ProviderReference,
    ResolvedExecutionConfig,
    RetryPolicy,
    TurnOperation,
)
from nexus_mcp.exceptions import ParseError, RetryableError, SubprocessError, SubprocessTimeoutError
from nexus_mcp.legacy.runner_backend import LegacyRunnerBackend, legacy_backends
from nexus_mcp.runners.factory import RunnerFactory
from tests.fixtures import (
    make_agent_job,
    make_agent_response,
    make_requested_config,
    make_workspace,
)


class RecordingContext:
    """Record normalized effects emitted by one legacy execution."""

    def __init__(self, *, resolved_config: ResolvedExecutionConfig | None = None) -> None:
        self.job = make_agent_job()
        self.attempt = JobAttempt(job_id=self.job.job_id, attempt_number=1)
        self.workspace = make_workspace()
        self.resolved_config = resolved_config or ResolvedExecutionConfig()
        self.events: list[BackendEvent] = []

    async def emit(self, event: BackendEvent) -> None:
        self.events.append(event)

    async def emit_output_delta(self, text: str) -> None:
        raise AssertionError("legacy runners do not emit output deltas")

    async def record_provider_reference(self, reference: ProviderReference) -> None:
        raise AssertionError("legacy runners do not expose provider references")

    async def request_input(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("legacy runners do not support input requests")

    async def wait_for_control(self):  # type: ignore[no-untyped-def]
        raise AssertionError("legacy runners do not support control signals")

    async def checkpoint(self) -> None:
        return


def test_legacy_backend_advertises_only_turn():
    backend = LegacyRunnerBackend("codex")

    assert backend.descriptor.capabilities.operations == frozenset({"turn"})
    assert backend.descriptor.capabilities.session_fork is False
    assert backend.descriptor.capabilities.cancellation is False
    assert backend.descriptor.capabilities.graceful_interrupt is False


@pytest.mark.parametrize(
    ("backend_id", "sandbox_modes"),
    [
        ("claude", frozenset({"danger_full_access"})),
        ("codex", frozenset({"danger_full_access"})),
        ("opencode", frozenset()),
        ("opencode_server", frozenset()),
    ],
)
def test_legacy_sandbox_modes_match_runner_support(backend_id: str, sandbox_modes: frozenset[str]):
    assert LegacyRunnerBackend(backend_id).descriptor.capabilities.sandbox_modes == sandbox_modes


def test_legacy_backends_and_default_manager_are_deterministic():
    backends = legacy_backends()
    expected = ["claude", "codex", "opencode", "opencode_server"]

    assert [backend.descriptor.backend_id for backend in backends] == expected
    manager = BackendManager()
    assert [manager.get(backend_id).descriptor.backend_id for backend_id in expected] == expected


async def test_legacy_backend_forces_single_runner_attempt_and_maps_turn():
    resolved = ResolvedExecutionConfig(
        model="model-test",
        timeout_seconds=25,
        output_limit_bytes=4096,
        retry_policy=RetryPolicy(max_attempts=8, base_delay_seconds=2, max_delay_seconds=7),
    )
    context = RecordingContext(resolved_config=resolved)
    runner = AsyncMock()
    runner.run.return_value = make_agent_response(output="done")
    operation = TurnOperation(
        prompt="hello",
        context={"label": "bridge-test"},
        file_refs=("README.md",),
    )

    with patch.object(RunnerFactory, "create", return_value=runner):
        result = await LegacyRunnerBackend("codex").execute(operation, context)

    request = runner.run.call_args.args[0]
    assert request.cli == "codex"
    assert request.prompt == "hello"
    assert request.context == {"label": "bridge-test"}
    assert request.file_refs == ["README.md"]
    assert request.model == "model-test"
    assert request.timeout == 25
    assert request.output_limit == 4096
    assert request.max_retries == 1
    assert request.retry_base_delay == 2
    assert request.retry_max_delay == 7
    assert request.cwd == context.workspace.canonical_path
    assert request.execution_mode == "default"
    assert result.message == "done"


async def test_legacy_availability_is_observed_without_changing_registration():
    with (
        patch(
            "nexus_mcp.legacy.runner_backend.detect_cli",
            return_value=CLIInfo(found=False),
        ),
        patch.object(RunnerFactory, "create") as create,
    ):
        availability = await LegacyRunnerBackend("codex").check_availability(make_workspace())

    assert availability.available is False
    assert availability.reason == "codex is not available"
    assert [backend.descriptor.backend_id for backend in legacy_backends()] == [
        "claude",
        "codex",
        "opencode",
        "opencode_server",
    ]
    create.assert_not_called()


async def test_legacy_version_detection_does_not_block_event_loop():
    marker_reached = asyncio.Event()

    async def mark_progress() -> None:
        await asyncio.sleep(0)
        marker_reached.set()

    def blocking_version_detection(backend_id: str) -> str:
        assert backend_id == "codex"
        time.sleep(0.05)
        return "1.2.3"

    marker_task = asyncio.create_task(mark_progress())
    with (
        patch(
            "nexus_mcp.legacy.runner_backend.detect_cli",
            return_value=CLIInfo(found=True, path="/usr/bin/codex"),
        ),
        patch(
            "nexus_mcp.legacy.runner_backend.get_cli_version",
            side_effect=blocking_version_detection,
        ),
    ):
        availability = await LegacyRunnerBackend("codex").check_availability(make_workspace())

    assert marker_reached.is_set()
    assert availability.version == "1.2.3"
    await marker_task


async def test_legacy_backend_normalizes_log_and_progress_events():
    context = RecordingContext()
    runner = AsyncMock()

    async def run(request, emitter, progress):  # type: ignore[no-untyped-def]
        await emitter("warning", "provider warning")
        await progress(2, 5, "Executing")
        return make_agent_response(output="done")

    runner.run.side_effect = run

    with patch.object(RunnerFactory, "create", return_value=runner):
        await LegacyRunnerBackend("codex").execute(TurnOperation(prompt="hello"), context)

    assert [(event.type, dict(event.payload)) for event in context.events] == [
        ("log", {"level": "warning", "message": "provider warning"}),
        ("progress", {"progress": 2.0, "total": 5.0, "message": "Executing"}),
    ]


async def test_legacy_backend_maps_truthful_yolo_pair():
    context = RecordingContext(
        resolved_config=ResolvedExecutionConfig(
            sandbox="danger_full_access",
            approval_policy="never",
        )
    )
    runner = AsyncMock()
    runner.run.return_value = make_agent_response(output="done")

    with patch.object(RunnerFactory, "create", return_value=runner):
        await LegacyRunnerBackend("codex").execute(TurnOperation(prompt="hello"), context)

    assert runner.run.call_args.args[0].execution_mode == "yolo"


@pytest.mark.parametrize(
    ("backend_id", "sandbox", "approval_policy"),
    [
        ("codex", "read_only", "provider_default"),
        ("codex", "workspace_write", "on_request"),
        ("codex", "danger_full_access", "provider_default"),
        ("codex", None, "never"),
        ("opencode", "danger_full_access", "never"),
    ],
)
async def test_legacy_backend_rejects_unadvertised_execution_policy(
    backend_id: str,
    sandbox: str | None,
    approval_policy: str,
):
    context = RecordingContext(
        resolved_config=ResolvedExecutionConfig(
            sandbox=sandbox,  # type: ignore[arg-type]
            approval_policy=approval_policy,  # type: ignore[arg-type]
        )
    )

    with (
        patch.object(RunnerFactory, "create") as create,
        pytest.raises(BackendFailure) as raised,
    ):
        await LegacyRunnerBackend(backend_id).execute(TurnOperation(prompt="hello"), context)

    assert raised.value.error.code == "unsupported_capability"
    assert raised.value.retry_disposition == "terminal"
    create.assert_not_called()


async def test_legacy_backend_uses_admitted_workspace(tmp_path: Path):
    context = RecordingContext()
    context.workspace = make_workspace(canonical_path=tmp_path)
    runner = AsyncMock()
    runner.run.return_value = make_agent_response(output="done")

    with patch.object(RunnerFactory, "create", return_value=runner):
        await LegacyRunnerBackend("codex").execute(TurnOperation(prompt="hello"), context)

    assert runner.run.call_args.args[0].cwd == tmp_path


async def test_opencode_server_rejects_non_process_workspace(tmp_path: Path):
    context = RecordingContext()
    context.workspace = make_workspace(canonical_path=tmp_path)

    with (
        patch.object(Path, "cwd", return_value=tmp_path / "different"),
        patch.object(RunnerFactory, "create") as create,
        pytest.raises(BackendFailure) as raised,
    ):
        await LegacyRunnerBackend("opencode_server").execute(TurnOperation(prompt="hello"), context)

    assert raised.value.error.code == "workspace_unsupported"
    assert raised.value.retry_disposition == "terminal"
    create.assert_not_called()


async def test_legacy_defaults_are_final_fallbacks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "nexus_mcp.legacy.runner_backend.get_runner_defaults",
        lambda _: type(
            "Defaults",
            (),
            {
                "model": "legacy-env-model",
                "timeout": 91,
                "output_limit": 8192,
                "max_retries": 3,
                "retry_base_delay": 4.0,
                "retry_max_delay": 12.0,
            },
        )(),
    )
    workspace = make_workspace()

    fallback = await LegacyRunnerBackend("codex").resolve_execution_config(
        make_requested_config(), workspace
    )
    overridden = await LegacyRunnerBackend("codex").resolve_execution_config(
        make_requested_config(workspace={"model": "workspace-model"}), workspace
    )

    assert fallback.model == "legacy-env-model"
    assert fallback.timeout_seconds == 91
    assert fallback.output_limit_bytes == 8192
    assert fallback.retry_policy == RetryPolicy(
        max_attempts=3,
        base_delay_seconds=4,
        max_delay_seconds=12,
    )
    assert fallback.sources == {
        "model": "legacy_nexus_fallback",
        "timeout_seconds": "legacy_nexus_fallback",
        "output_limit_bytes": "legacy_nexus_fallback",
        "retry_policy": "legacy_nexus_fallback",
    }
    assert overridden.model == "workspace-model"
    assert overridden.sources["model"] == "workspace"
    assert "provider" not in overridden.sources.values()


@pytest.mark.parametrize(
    ("error", "code", "disposition", "recoverable"),
    [
        (RetryableError("retry", returncode=429), "provider_failed", "safe_to_retry", True),
        (
            ParseError("invalid", raw_output="secret"),
            "structured_output_invalid",
            "terminal",
            False,
        ),
        (SubprocessError("failed", stderr="secret"), "provider_failed", "terminal", False),
        (
            SubprocessTimeoutError("late", timeout=30, stderr="secret"),
            "timeout",
            "terminal",
            False,
        ),
        (RuntimeError("unexpected secret"), "internal_error", "terminal", False),
    ],
)
async def test_legacy_exception_mapping_is_normalized_and_bounded(
    error: Exception,
    code: str,
    disposition: str,
    recoverable: bool,
):
    runner = AsyncMock()
    runner.run.side_effect = error

    with (
        patch.object(RunnerFactory, "create", return_value=runner),
        pytest.raises(BackendFailure) as raised,
    ):
        await LegacyRunnerBackend("codex").execute(
            TurnOperation(prompt="hello"), RecordingContext()
        )

    assert raised.value.error.code == code
    assert raised.value.retry_disposition == disposition
    assert raised.value.error.retry_disposition == disposition
    assert raised.value.error.recoverable is recoverable
    assert raised.value.error.details["legacy_exception_type"] == type(error).__name__
    assert "secret" not in raised.value.error.message


@pytest.mark.parametrize("exception_type", ["Bad-Name", "A" * 129])
async def test_legacy_exception_type_detail_is_a_bounded_identifier(exception_type: str):
    invalid_exception = type(exception_type, (Exception,), {})("secret")
    runner = AsyncMock()
    runner.run.side_effect = invalid_exception

    with (
        patch.object(RunnerFactory, "create", return_value=runner),
        pytest.raises(BackendFailure) as raised,
    ):
        await LegacyRunnerBackend("codex").execute(
            TurnOperation(prompt="hello"), RecordingContext()
        )

    normalized_type = raised.value.error.details["legacy_exception_type"]
    assert normalized_type == "Exception"
    assert isinstance(normalized_type, str)
    assert normalized_type.isidentifier()
    assert len(normalized_type) <= 128


async def test_legacy_reconcile_is_unknown_without_runner_replay():
    context = RecordingContext()

    with patch.object(RunnerFactory, "create") as create:
        outcome = await LegacyRunnerBackend("codex").reconcile((), context)

    assert outcome.kind == "unknown"
    assert outcome.error.code == "outcome_unknown"
    assert outcome.error.recoverable is True
    create.assert_not_called()
