"""Runtime backend registration tests."""

from nexus_mcp.backends.claude_agent import ClaudeAgentBackend
from nexus_mcp.legacy import LegacyRunnerBackend
from nexus_mcp.mcp.runtime import select_backends

EXPECTED_IDS = ["claude", "codex", "opencode", "opencode_server"]


def test_native_claude_is_default(monkeypatch):
    """The default runtime substitutes the SDK implementation for Claude."""
    monkeypatch.delenv("NEXUS_ENABLE_LEGACY_RUNNERS", raising=False)

    backends = {backend.descriptor.backend_id: backend for backend in select_backends()}

    assert sorted(backends) == EXPECTED_IDS
    assert isinstance(backends["claude"], ClaudeAgentBackend)
    assert all(
        isinstance(backends[name], LegacyRunnerBackend) for name in EXPECTED_IDS if name != "claude"
    )


def test_flag_restores_legacy_claude(monkeypatch):
    """The compatibility flag preserves all legacy runner adapters."""
    monkeypatch.setenv("NEXUS_ENABLE_LEGACY_RUNNERS", "1")

    backends = select_backends()

    assert sorted(backend.descriptor.backend_id for backend in backends) == EXPECTED_IDS
    assert all(isinstance(backend, LegacyRunnerBackend) for backend in backends)
