"""E2E compatibility contracts for legacy prompt aliases backed by Nexus jobs."""

import asyncio
import json
import os
import sqlite3
from contextlib import asynccontextmanager

import pytest

from nexus_mcp.mcp.runtime import MCPRuntime, runtime_provider
from nexus_mcp.server import batch_prompt, prompt
from tests.fakes import FakeRunner
from tests.fixtures import CODEX_NDJSON_RESPONSE, create_mock_process, strip_runner_header


def _job_rows(*columns: str) -> list[tuple[object, ...]]:
    database_path = os.environ["NEXUS_DB_PATH"]
    with sqlite3.connect(database_path) as connection:
        query = f"SELECT {', '.join(columns)} FROM jobs ORDER BY created_at_ms"
        return connection.execute(query).fetchall()


@pytest.mark.e2e
async def test_single_prompt_submits_nexus_job_and_preserves_header(
    job_mcp_client, fake_runner_registry
):
    result = await job_mcp_client.call_tool(
        "prompt",
        {
            "cli": fake_runner_registry,
            "prompt": "compatibility prompt",
            "context": {"fake_output": "compatibility output"},
        },
    )

    assert result.is_error is False
    assert result.data == "[cli: fake | model: default | mode: default]\n\ncompatibility output"
    rows = _job_rows("backend_id", "operation_kind", "operation_json", "state")
    assert len(rows) == 1
    backend_id, operation_kind, operation_json, state = rows[0]
    assert backend_id == fake_runner_registry
    assert operation_kind == "turn"
    assert json.loads(str(operation_json))["prompt"] == "compatibility prompt"
    assert state == "completed"


@pytest.mark.e2e
async def test_task_true_keeps_docket_id_out_of_nexus_jobs(job_mcp_client, fake_runner_registry):
    task = await job_mcp_client.call_tool(
        "prompt",
        {
            "cli": fake_runner_registry,
            "prompt": "background compatibility prompt",
            "context": {"fake_output": "background output"},
        },
        task=True,
    )
    result = await task

    assert result.is_error is False
    assert strip_runner_header(result.data) == "background output"
    nexus_job_ids = {str(row[0]) for row in _job_rows("job_id")}
    assert len(nexus_job_ids) == 1
    assert task.task_id not in nexus_job_ids


@pytest.mark.e2e
async def test_batch_jobs_preserve_order_and_unique_labels(job_mcp_client, fake_runner_registry):
    result = await job_mcp_client.call_tool(
        "batch_prompt",
        {
            "tasks": [
                {
                    "cli": fake_runner_registry,
                    "prompt": "first",
                    "context": {"fake_output": "first output"},
                },
                {
                    "cli": fake_runner_registry,
                    "prompt": "second",
                    "context": {"fake_output": "second output"},
                },
            ]
        },
    )

    assert result.is_error is False
    assert result.data.succeeded == 2
    assert [item.label for item in result.data.results] == ["fake", "fake-2"]
    assert [strip_runner_header(item.output) for item in result.data.results] == [
        "first output",
        "second output",
    ]
    rows = _job_rows("job_id", "session_id")
    assert len(rows) == 2
    assert len({str(row[0]) for row in rows}) == 2
    assert len({str(row[1]) for row in rows}) == 2


@pytest.mark.e2e
async def test_batch_max_concurrency_two_overlaps_provider_execution(
    monkeypatch, fake_runner_registry, fast_job_runtime
):
    """Two admitted compatibility jobs reach their providers concurrently."""
    del fast_job_runtime
    original_run = FakeRunner.run
    active = 0
    maximum_active = 0
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def barrier_run(self, request, emitter=None, progress=None):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            both_started.set()
        try:
            await release.wait()
            return await original_run(self, request, emitter=emitter, progress=progress)
        finally:
            active -= 1

    monkeypatch.setattr(FakeRunner, "run", barrier_run)
    batch = asyncio.create_task(
        batch_prompt(
            tasks=[
                {"cli": fake_runner_registry, "prompt": "first"},
                {"cli": fake_runner_registry, "prompt": "second"},
            ],
            max_concurrency=2,
        )
    )
    try:
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
    finally:
        release.set()
        await batch

    assert maximum_active == 2


@pytest.mark.e2e
async def test_concurrent_singleton_calls_share_fixed_runtime_capacity(
    monkeypatch, fake_runner_registry, fast_job_runtime
):
    """Two singleton calls on one installed runtime can enter providers concurrently."""
    del fast_job_runtime
    original_run = FakeRunner.run
    active = 0
    maximum_active = 0
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def barrier_run(self, request, emitter=None, progress=None):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            both_started.set()
        try:
            await release.wait()
            return await original_run(self, request, emitter=emitter, progress=progress)
        finally:
            active -= 1

    monkeypatch.setattr(FakeRunner, "run", barrier_run)
    async with (
        MCPRuntime.open(runtime_provider.tuning) as runtime,
        runtime_provider.install(runtime),
    ):
        calls = [
            asyncio.create_task(
                batch_prompt(
                    tasks=[
                        {
                            "cli": fake_runner_registry,
                            "prompt": prompt_text,
                            "context": {"fake_output": f"{prompt_text} output"},
                        }
                    ],
                    max_concurrency=1,
                )
            )
            for prompt_text in ("first", "second")
        ]
        try:
            await asyncio.wait_for(both_started.wait(), timeout=1.0)
        finally:
            release.set()
            responses = await asyncio.gather(*calls)

    assert maximum_active == 2
    assert [response.succeeded for response in responses] == [1, 1]


@pytest.mark.e2e
async def test_batch_max_concurrency_one_serializes_provider_execution(
    monkeypatch, fake_runner_registry, fast_job_runtime
):
    """The compatibility semaphore still keeps provider execution serial at one."""
    del fast_job_runtime
    original_run = FakeRunner.run
    active = 0
    maximum_active = 0
    calls = 0
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def observed_run(self, request, emitter=None, progress=None):
        nonlocal active, maximum_active, calls
        calls += 1
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            if calls == 1:
                first_started.set()
                await release_first.wait()
            return await original_run(self, request, emitter=emitter, progress=progress)
        finally:
            active -= 1

    monkeypatch.setattr(FakeRunner, "run", observed_run)
    batch = asyncio.create_task(
        batch_prompt(
            tasks=[
                {"cli": fake_runner_registry, "prompt": "first"},
                {"cli": fake_runner_registry, "prompt": "second"},
            ],
            max_concurrency=1,
        )
    )
    await asyncio.wait_for(first_started.wait(), timeout=1.0)
    await asyncio.sleep(0)
    assert calls == 1
    release_first.set()
    await batch

    assert calls == 2
    assert maximum_active == 1


@pytest.mark.e2e
async def test_batch_partial_failure_preserves_legacy_error_type(
    mock_subprocess, fast_job_mcp_client
):
    def subprocess_for_prompt(*args, **_kwargs):
        command = list(args)
        prompt_text = str(command[command.index("exec") + 1])
        if "succeed" in prompt_text:
            return create_mock_process(stdout=CODEX_NDJSON_RESPONSE)
        return create_mock_process(stdout="not valid json", returncode=0)

    mock_subprocess.side_effect = subprocess_for_prompt
    result = await fast_job_mcp_client.call_tool(
        "batch_prompt",
        {
            "tasks": [
                {"cli": "codex", "prompt": "please succeed"},
                {"cli": "codex", "prompt": "will fail"},
            ]
        },
    )

    assert result.is_error is False
    assert result.data.succeeded == 1
    assert result.data.failed == 1
    failed = next(item for item in result.data.results if item.error is not None)
    assert failed.error_type == "ParseError"
    assert failed.error == "Legacy backend codex returned an invalid response"


@pytest.mark.e2e
async def test_explicit_model_and_yolo_mode_are_admitted_truthfully(
    mock_subprocess, fast_job_mcp_client
):
    mock_subprocess.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE)

    result = await fast_job_mcp_client.call_tool(
        "prompt",
        {
            "cli": "codex",
            "prompt": "configured prompt",
            "model": "gpt-test",
            "execution_mode": "yolo",
        },
    )

    assert result.data.startswith("[cli: codex | model: gpt-test | mode: yolo]")
    requested_json, resolved_json = _job_rows("requested_config_json", "resolved_config_json")[0]
    explicit = json.loads(str(requested_json))["explicit"]
    resolved = json.loads(str(resolved_json))
    assert explicit["model"] == "gpt-test"
    assert explicit["sandbox"] == "danger_full_access"
    assert explicit["approval_policy"] == "never"
    assert resolved["sandbox"] == "danger_full_access"
    assert resolved["approval_policy"] == "never"
    args = list(mock_subprocess.call_args.args)
    assert "--model" in args
    assert "gpt-test" in args
    assert "--dangerously-bypass-approvals-and-sandbox" in args


@pytest.mark.e2e
async def test_legacy_preferences_feed_job_configuration(mock_subprocess, fast_job_mcp_client):
    mock_subprocess.return_value = create_mock_process(stdout=CODEX_NDJSON_RESPONSE)
    await fast_job_mcp_client.call_tool(
        "set_preferences",
        {
            "model": "preferred-model",
            "execution_mode": "yolo",
            "max_retries": 1,
            "timeout": 33,
        },
    )

    result = await fast_job_mcp_client.call_tool(
        "prompt", {"cli": "codex", "prompt": "use preferences"}
    )

    assert result.data.startswith("[cli: codex | model: preferred-model | mode: yolo]")
    explicit = json.loads(str(_job_rows("requested_config_json")[0][0]))["explicit"]
    assert explicit["model"] == "preferred-model"
    assert explicit["sandbox"] == "danger_full_access"
    assert explicit["approval_policy"] == "never"
    assert explicit["retry_policy"]["max_attempts"] == 1
    assert explicit["timeout_seconds"] == 33


@pytest.mark.e2e
async def test_inherited_yolo_remains_compatibility_only_for_opencode(
    mock_subprocess, fast_job_mcp_client
):
    """An inherited yolo header does not request unsupported OpenCode sandbox policy."""
    from tests.fixtures import OPENCODE_NDJSON_RESPONSE

    mock_subprocess.return_value = create_mock_process(stdout=OPENCODE_NDJSON_RESPONSE)
    await fast_job_mcp_client.call_tool("set_preferences", {"execution_mode": "yolo"})

    result = await fast_job_mcp_client.call_tool(
        "prompt", {"cli": "opencode", "prompt": "use inherited preference"}
    )

    assert result.is_error is False
    assert result.data.startswith("[cli: opencode | model: default | mode: yolo]")
    explicit = json.loads(str(_job_rows("requested_config_json")[0][0]))["explicit"]
    assert explicit["sandbox"] is None
    assert explicit["approval_policy"] is None


@pytest.mark.e2e
async def test_direct_prompt_opens_and_closes_one_temporary_runtime(
    monkeypatch, fake_runner_registry, fast_job_runtime
):
    del fast_job_runtime
    lifecycle: list[str] = []
    original_open = MCPRuntime.open

    @asynccontextmanager
    async def tracked_open(tuning=None):
        lifecycle.append("open")
        async with original_open(tuning) as runtime:
            yield runtime
        lifecycle.append("close")

    monkeypatch.setattr(MCPRuntime, "open", staticmethod(tracked_open))

    result = await prompt(
        cli=fake_runner_registry,
        prompt="direct compatibility prompt",
        context={"fake_output": "direct output"},
    )

    assert strip_runner_header(result) == "direct output"
    assert lifecycle == ["open", "close"]
