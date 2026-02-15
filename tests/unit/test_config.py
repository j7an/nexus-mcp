"""Tests for configuration module (environment variable settings)."""

import os
from unittest.mock import patch

import pytest

from nexus_mcp.config import get_agent_env, get_global_output_limit, get_global_timeout
from nexus_mcp.exceptions import ConfigurationError


class TestGetGlobalOutputLimit:
    """Test get_global_output_limit() function."""

    def test_get_global_output_limit_default(self):
        """Output limit defaults to 50KB if env var not set."""
        limit = get_global_output_limit()
        assert limit == 50_000  # 50KB default

    @patch.dict(os.environ, {"NEXUS_OUTPUT_LIMIT_BYTES": "100000"})
    def test_get_global_output_limit_from_env(self):
        """Output limit can be overridden via NEXUS_OUTPUT_LIMIT_BYTES."""
        limit = get_global_output_limit()
        assert limit == 100_000

    @patch.dict(os.environ, {"NEXUS_OUTPUT_LIMIT_BYTES": "not-a-number"})
    def test_get_global_output_limit_invalid_raises_configuration_error(self):
        """Invalid output limit value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_global_output_limit()

        assert exc_info.value.config_key == "NEXUS_OUTPUT_LIMIT_BYTES"
        assert "Invalid output limit value: 'not-a-number'" in str(exc_info.value)


class TestGetGlobalTimeout:
    """Test get_global_timeout() function."""

    def test_get_global_timeout_default(self):
        """Timeout defaults to 600s if env var not set."""
        timeout = get_global_timeout()
        assert timeout == 600  # Match existing process.py default

    @patch.dict(os.environ, {"NEXUS_TIMEOUT_SECONDS": "300"})
    def test_get_global_timeout_from_env(self):
        """Timeout can be overridden via NEXUS_TIMEOUT_SECONDS."""
        timeout = get_global_timeout()
        assert timeout == 300

    @patch.dict(os.environ, {"NEXUS_TIMEOUT_SECONDS": "abc"})
    def test_get_global_timeout_invalid_raises_configuration_error(self):
        """Invalid timeout value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_global_timeout()

        assert exc_info.value.config_key == "NEXUS_TIMEOUT_SECONDS"
        assert "Invalid timeout value: 'abc'" in str(exc_info.value)


class TestGetAgentEnv:
    """Test get_agent_env() function."""

    @patch.dict(os.environ, {"NEXUS_GEMINI_PATH": "/custom/gemini"})
    def test_get_agent_env_returns_value(self):
        """get_agent_env reads NEXUS_{AGENT}_{KEY} env vars."""
        path = get_agent_env("gemini", "PATH")
        assert path == "/custom/gemini"

    def test_get_agent_env_returns_default(self):
        """get_agent_env returns default if env var not set."""
        path = get_agent_env("codex", "PATH", default="/usr/bin/codex")
        assert path == "/usr/bin/codex"

    @patch.dict(os.environ, {"NEXUS_GEMINI_MODEL": "gemini-2.5-flash"})
    def test_get_agent_env_case_insensitive(self):
        """get_agent_env normalizes agent name to uppercase."""
        model = get_agent_env("gemini", "MODEL")
        assert model == "gemini-2.5-flash"
