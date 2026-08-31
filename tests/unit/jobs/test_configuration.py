"""Immutable configuration snapshots captured at Nexus admission."""

import hashlib
from pathlib import Path

import pytest

from nexus_mcp.core.models import (
    ConfigLayerSnapshot,
    ExecutionConfigValues,
    RequestedExecutionConfig,
    ResolvedExecutionConfig,
    RetryPolicy,
    Workspace,
)
from nexus_mcp.exceptions import ConfigurationError
from nexus_mcp.jobs import configuration
from nexus_mcp.jobs.configuration import (
    NexusConfigResolver,
    read_config_file,
    user_config_path,
    workspace_config_path,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Provide a durable workspace with a canonical temporary location."""
    path = tmp_path / "workspace"
    path.mkdir()
    return Workspace(workspace_id="workspace-test", canonical_path=path)


@pytest.fixture(autouse=True)
def clear_task_nine_environment(monkeypatch: pytest.MonkeyPatch):
    """Keep admission-snapshot tests independent from the developer environment."""
    for name in (
        "NEXUS_CONFIG_PATH",
        "NEXUS_TIMEOUT_SECONDS",
        "NEXUS_OUTPUT_LIMIT_BYTES",
        "NEXUS_RETRY_MAX_ATTEMPTS",
        "NEXUS_RETRY_BASE_DELAY",
        "NEXUS_RETRY_MAX_DELAY",
        "NEXUS_CODEX_MODEL",
        "NEXUS_CODEX_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_linux_user_config_path_honors_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Linux configuration follows XDG_CONFIG_HOME when no override exists."""
    monkeypatch.delenv("NEXUS_CONFIG_PATH", raising=False)
    monkeypatch.setattr(configuration.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert user_config_path() == tmp_path / "nexus-mcp" / "config.toml"


def test_user_config_path_honors_explicit_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """NEXUS_CONFIG_PATH wins over every platform-specific default."""
    override = tmp_path / "chosen.toml"
    monkeypatch.setenv("NEXUS_CONFIG_PATH", str(override))
    monkeypatch.setattr(configuration.sys, "platform", "darwin")

    assert user_config_path() == override


@pytest.mark.parametrize(
    ("platform", "environment_name", "expected"),
    [
        ("darwin", None, Path("Library/Application Support/nexus-mcp/config.toml")),
        ("linux", None, Path(".config/nexus-mcp/config.toml")),
        ("win32", "APPDATA", Path("NexusAppData/nexus-mcp/config.toml")),
    ],
)
def test_user_config_path_uses_platform_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    environment_name: str | None,
    expected: Path,
):
    """Each supported platform has the documented user configuration location."""
    monkeypatch.delenv("NEXUS_CONFIG_PATH", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(configuration.sys, "platform", platform)
    monkeypatch.setattr(configuration.Path, "home", lambda: tmp_path)
    if environment_name is not None:
        monkeypatch.setenv(environment_name, str(tmp_path / "NexusAppData"))

    assert user_config_path() == tmp_path / expected


def test_workspace_config_path_is_scoped_to_canonical_workspace(workspace: Workspace):
    """Workspace configuration cannot be redirected outside its durable workspace."""
    assert workspace_config_path(workspace) == workspace.canonical_path / ".nexus" / "config.toml"


@pytest.mark.parametrize(
    "content",
    [
        'unsupported = "value"\n',
        '[defaults]\napi_key = "not-allowed"\n',
        '[backends.codex]\nauthorization = "not-allowed"\n',
    ],
)
def test_unknown_or_secret_keys_are_rejected(tmp_path: Path, content: str):
    """Only the credential-free TOML schema is admissible."""
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError):
        read_config_file(path, backend_id="codex")


def test_backend_section_overrides_defaults_within_one_file(tmp_path: Path):
    """Backend-specific values win without discarding untouched defaults."""
    path = tmp_path / "config.toml"
    path.write_text(
        """
[defaults]
model = "default-model"
timeout_seconds = 30
max_attempts = 2
retry_base_delay_seconds = 1.5
retry_max_delay_seconds = 20.0

[backends.codex]
model = "codex-model"
max_attempts = 4
""".strip(),
        encoding="utf-8",
    )

    values = read_config_file(path, backend_id="codex")

    assert values == ExecutionConfigValues(
        model="codex-model",
        timeout_seconds=30,
        retry_policy=RetryPolicy(max_attempts=4, base_delay_seconds=1.5, max_delay_seconds=20.0),
    )


def test_resolution_precedence_is_explicit_provider_workspace_user_environment():
    """Task 1 resolution remains explicit, provider, workspace, user, environment."""
    requested = RequestedExecutionConfig(
        explicit=ExecutionConfigValues(timeout_seconds=10),
        workspace=ConfigLayerSnapshot(
            values=ExecutionConfigValues(model="workspace-model", timeout_seconds=20),
            source="workspace",
            source_hash="a" * 64,
        ),
        user=ConfigLayerSnapshot(
            values=ExecutionConfigValues(model="user-model"),
            source="user",
            source_hash="b" * 64,
        ),
        environment=ConfigLayerSnapshot(
            values=ExecutionConfigValues(model="env-model"),
            source="environment",
            source_hash="c" * 64,
        ),
    )

    resolved = ResolvedExecutionConfig.from_requested(
        requested, backend_defaults=ExecutionConfigValues(model="provider-model")
    )

    assert resolved.timeout_seconds == 10
    assert resolved.model == "provider-model"


def test_snapshot_captures_workspace_user_and_environment_layers(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Workspace,
    tmp_path: Path,
):
    """Admission retains each lower-precedence source and its normalized values."""
    workspace_path = workspace_config_path(workspace)
    workspace_path.parent.mkdir()
    workspace_path.write_text("[defaults]\ntimeout_seconds = 20\n", encoding="utf-8")
    user_path = tmp_path / "user-config.toml"
    user_contents = "[defaults]\noutput_limit_bytes = 4000\n"
    user_path.write_text(user_contents, encoding="utf-8")
    monkeypatch.setenv("NEXUS_CODEX_MODEL", "environment-model")

    requested = NexusConfigResolver(user_path=user_path).snapshot(
        "codex", workspace, ExecutionConfigValues(timeout_seconds=10)
    )

    assert requested.explicit == ExecutionConfigValues(timeout_seconds=10)
    assert requested.workspace is not None
    assert requested.workspace.values == ExecutionConfigValues(timeout_seconds=20)
    assert requested.workspace.source == str(workspace_path)
    assert requested.user is not None
    assert requested.user.values == ExecutionConfigValues(output_limit_bytes=4000)
    assert requested.user.source_hash == hashlib.sha256(user_contents.encode()).hexdigest()
    assert requested.environment is not None
    assert requested.environment.values == ExecutionConfigValues(model="environment-model")
    assert requested.environment.source == "environment"
    assert len(requested.environment.source_hash) == 64


def test_environment_mapping_captures_only_supported_execution_values(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Workspace,
):
    """The environment layer maps each documented setting and no extension keys."""
    monkeypatch.setenv("NEXUS_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("NEXUS_OUTPUT_LIMIT_BYTES", "50000")
    monkeypatch.setenv("NEXUS_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("NEXUS_RETRY_BASE_DELAY", "1.0")
    monkeypatch.setenv("NEXUS_RETRY_MAX_DELAY", "30.0")
    monkeypatch.setenv("NEXUS_CODEX_MODEL", "captured-model")
    monkeypatch.setenv("NEXUS_CODEX_API_KEY", "never-captured")

    requested = NexusConfigResolver(user_path=workspace.canonical_path / "missing.toml").snapshot(
        "codex", workspace, ExecutionConfigValues()
    )

    assert requested.environment is not None
    assert requested.environment.values == ExecutionConfigValues(
        model="captured-model",
        timeout_seconds=600,
        output_limit_bytes=50000,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=1.0, max_delay_seconds=30.0),
    )
    assert "NEXUS" not in requested.environment.model_dump_json()


def test_invalid_environment_value_is_rejected_by_execution_config_schema(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Workspace,
):
    """Invalid captured values cannot enter an immutable request snapshot."""
    monkeypatch.setenv("NEXUS_TIMEOUT_SECONDS", "0")

    with pytest.raises(ConfigurationError) as error:
        NexusConfigResolver(user_path=workspace.canonical_path / "missing.toml").snapshot(
            "codex", workspace, ExecutionConfigValues()
        )

    assert error.value.config_key == "NEXUS_TIMEOUT_SECONDS"


def test_invalid_retry_environment_value_identifies_the_rejected_setting(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Workspace,
):
    """A rejected nested retry setting identifies its own input rather than another retry variable.

    This prevents a valid sibling retry variable from being blamed for a bad value.
    """
    monkeypatch.setenv("NEXUS_RETRY_MAX_ATTEMPTS", "0")
    monkeypatch.setenv("NEXUS_RETRY_MAX_DELAY", "30.0")

    with pytest.raises(ConfigurationError) as error:
        NexusConfigResolver(user_path=workspace.canonical_path / "missing.toml").snapshot(
            "codex", workspace, ExecutionConfigValues()
        )

    assert error.value.config_key == "NEXUS_RETRY_MAX_ATTEMPTS"


def test_snapshot_does_not_change_when_file_changes(
    workspace: Workspace,
    tmp_path: Path,
):
    """A request retains its captured data after its source file changes."""
    config_path = workspace_config_path(workspace)
    config_path.parent.mkdir()
    config_path.write_text("[defaults]\ntimeout_seconds = 30\n", encoding="utf-8")
    resolver = NexusConfigResolver(user_path=tmp_path / "missing.toml")

    first = resolver.snapshot("codex", workspace, ExecutionConfigValues())
    config_path.write_text("[defaults]\ntimeout_seconds = 99\n", encoding="utf-8")

    assert first.workspace is not None
    assert first.workspace.values.timeout_seconds == 30
    assert first.workspace.values.timeout_seconds != 99


def test_snapshot_defers_provider_native_values_until_worker_resolution(
    workspace: Workspace,
    tmp_path: Path,
):
    """Admission does not invent provider defaults before the worker resolves them."""
    requested = NexusConfigResolver(user_path=tmp_path / "missing.toml").snapshot(
        "codex", workspace, ExecutionConfigValues()
    )

    assert requested.explicit.model is None
    resolved = ResolvedExecutionConfig.from_requested(
        requested, backend_defaults=ExecutionConfigValues(model="provider-model")
    )
    assert resolved.model == "provider-model"
