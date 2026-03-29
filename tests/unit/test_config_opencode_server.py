"""Tests for opencode_server connection config env vars."""

import pytest

from nexus_mcp.config_resolver import get_opencode_server_auth, get_opencode_server_url


class TestOpenCodeServerUrl:
    def test_default_url(self, monkeypatch):
        monkeypatch.delenv("NEXUS_OPENCODE_SERVER_URL", raising=False)
        assert get_opencode_server_url() == "http://localhost:4096"

    def test_custom_url(self, monkeypatch):
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://myhost:8080")
        assert get_opencode_server_url() == "http://myhost:8080"

    def test_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_URL", "http://localhost:4096/")
        assert get_opencode_server_url() == "http://localhost:4096"


class TestOpenCodeServerAuth:
    def test_default_username(self, monkeypatch):
        monkeypatch.delenv("NEXUS_OPENCODE_SERVER_USERNAME", raising=False)
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "secret")
        username, password = get_opencode_server_auth()
        assert username == "opencode"
        assert password == "secret"

    def test_custom_username(self, monkeypatch):
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_USERNAME", "admin")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_PASSWORD", "secret")
        username, password = get_opencode_server_auth()
        assert username == "admin"
        assert password == "secret"

    def test_missing_password_raises(self, monkeypatch):
        monkeypatch.delenv("NEXUS_OPENCODE_SERVER_PASSWORD", raising=False)
        with pytest.raises(ValueError, match="NEXUS_OPENCODE_SERVER_PASSWORD"):
            get_opencode_server_auth()


class TestRunnerDefaults:
    def test_operational_defaults_work_for_opencode_server(self, monkeypatch):
        """Existing get_runner_defaults works for opencode_server via NEXUS_OPENCODE_SERVER_*."""
        from nexus_mcp.config_resolver import get_runner_defaults

        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_TIMEOUT", "300")
        monkeypatch.setenv("NEXUS_OPENCODE_SERVER_MODEL", "kimi-k2.5")
        defaults = get_runner_defaults("opencode_server")
        assert defaults.timeout == 300
        assert defaults.model == "kimi-k2.5"
