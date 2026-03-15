"""Tests for configuration module (environment variable settings)."""

import os
from unittest.mock import patch

import pytest

from nexus_mcp.config import (
    get_agent_env,
    get_cli_detection_timeout,
    get_global_output_limit,
    get_global_timeout,
    get_retry_base_delay,
    get_retry_max_attempts,
    get_retry_max_delay,
    get_tool_timeout,
)
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


class TestGetRetryMaxAttempts:
    """Test get_retry_max_attempts() function."""

    def test_default_is_three(self):
        """Max attempts defaults to 3 if env var not set."""
        assert get_retry_max_attempts() == 3

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_ATTEMPTS": "5"})
    def test_custom_value_from_env(self):
        """Max attempts can be overridden via NEXUS_RETRY_MAX_ATTEMPTS."""
        assert get_retry_max_attempts() == 5

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_ATTEMPTS": "not-a-number"})
    def test_invalid_raises_configuration_error(self):
        """Invalid value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_retry_max_attempts()
        assert exc_info.value.config_key == "NEXUS_RETRY_MAX_ATTEMPTS"
        assert "Invalid retry max attempts value" in str(exc_info.value)


class TestGetRetryBaseDelay:
    """Test get_retry_base_delay() function."""

    def test_default_is_two_seconds(self):
        """Base delay defaults to 2.0s if env var not set."""
        assert get_retry_base_delay() == 2.0

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "0.5"})
    def test_custom_float_from_env(self):
        """Base delay accepts float values."""
        assert get_retry_base_delay() == 0.5

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "1"})
    def test_integer_string_accepted(self):
        """Integer string is coerced to float."""
        assert get_retry_base_delay() == 1.0

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "not-a-float"})
    def test_invalid_raises_configuration_error(self):
        """Invalid value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_retry_base_delay()
        assert exc_info.value.config_key == "NEXUS_RETRY_BASE_DELAY"
        assert "Invalid retry base delay value" in str(exc_info.value)

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "inf"})
    def test_inf_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="must be a finite number"):
            get_retry_base_delay()

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "nan"})
    def test_nan_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="must be a finite number"):
            get_retry_base_delay()

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "-1.0"})
    def test_negative_raises_configuration_error(self):
        """Negative base delay is semantically invalid and raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_retry_base_delay()
        assert exc_info.value.config_key == "NEXUS_RETRY_BASE_DELAY"
        assert "non-negative" in str(exc_info.value)

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "0.0"})
    def test_zero_base_delay_allowed(self):
        """Zero is allowed (useful for testing — sleep(0) returns immediately)."""
        assert get_retry_base_delay() == 0.0


class TestGetRetryMaxDelay:
    """Test get_retry_max_delay() function."""

    def test_default_is_sixty_seconds(self):
        """Max delay defaults to 60.0s if env var not set."""
        assert get_retry_max_delay() == 60.0

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_DELAY": "30.0"})
    def test_custom_float_from_env(self):
        """Max delay can be overridden via NEXUS_RETRY_MAX_DELAY."""
        assert get_retry_max_delay() == 30.0

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_DELAY": "bad"})
    def test_invalid_raises_configuration_error(self):
        """Invalid value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_retry_max_delay()
        assert exc_info.value.config_key == "NEXUS_RETRY_MAX_DELAY"
        assert "Invalid retry max delay value" in str(exc_info.value)

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_DELAY": "inf"})
    def test_inf_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="must be a finite number"):
            get_retry_max_delay()

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_DELAY": "nan"})
    def test_nan_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="must be a finite number"):
            get_retry_max_delay()

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_DELAY": "-5.0"})
    def test_negative_raises_configuration_error(self):
        """Negative max delay is semantically invalid and raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_retry_max_delay()
        assert exc_info.value.config_key == "NEXUS_RETRY_MAX_DELAY"
        assert "non-negative" in str(exc_info.value)

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_DELAY": "0.0"})
    def test_zero_max_delay_allowed(self):
        """Zero is allowed — means no cap on wait time, which is unusual but valid."""
        assert get_retry_max_delay() == 0.0


class TestGetToolTimeout:
    """Test get_tool_timeout() function."""

    def test_default_is_900(self, monkeypatch):
        """Tool timeout defaults to 900.0s (15 min) if env var not set."""
        monkeypatch.delenv("NEXUS_TOOL_TIMEOUT_SECONDS", raising=False)
        assert get_tool_timeout() == 900.0

    @patch.dict(os.environ, {"NEXUS_TOOL_TIMEOUT_SECONDS": "1800"})
    def test_custom_value_from_env(self):
        """Tool timeout can be overridden via NEXUS_TOOL_TIMEOUT_SECONDS."""
        assert get_tool_timeout() == 1800.0

    @patch.dict(os.environ, {"NEXUS_TOOL_TIMEOUT_SECONDS": "0"})
    def test_zero_returns_none(self):
        """NEXUS_TOOL_TIMEOUT_SECONDS=0 disables the timeout (returns None)."""
        assert get_tool_timeout() is None

    @patch.dict(os.environ, {"NEXUS_TOOL_TIMEOUT_SECONDS": "-5"})
    def test_negative_raises(self):
        """Negative timeout value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_tool_timeout()
        assert exc_info.value.config_key == "NEXUS_TOOL_TIMEOUT_SECONDS"
        assert "must be non-negative" in str(exc_info.value)

    @patch.dict(os.environ, {"NEXUS_TOOL_TIMEOUT_SECONDS": "abc"})
    def test_invalid_raises(self):
        """Non-numeric value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_tool_timeout()
        assert exc_info.value.config_key == "NEXUS_TOOL_TIMEOUT_SECONDS"

    @patch.dict(os.environ, {"NEXUS_TOOL_TIMEOUT_SECONDS": "nan"})
    def test_nan_raises(self):
        """NaN value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_tool_timeout()
        assert exc_info.value.config_key == "NEXUS_TOOL_TIMEOUT_SECONDS"
        assert "must be a finite number" in str(exc_info.value)

    @patch.dict(os.environ, {"NEXUS_TOOL_TIMEOUT_SECONDS": "inf"})
    def test_inf_raises(self):
        """Infinite value should raise ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_tool_timeout()
        assert exc_info.value.config_key == "NEXUS_TOOL_TIMEOUT_SECONDS"
        assert "must be a finite number" in str(exc_info.value)

    @patch.dict(os.environ, {"NEXUS_TOOL_TIMEOUT_SECONDS": "-inf"})
    def test_negative_inf_raises(self):
        """Negative infinity is rejected by the finiteness check."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_tool_timeout()
        assert exc_info.value.config_key == "NEXUS_TOOL_TIMEOUT_SECONDS"
        assert "must be a finite number" in str(exc_info.value)


class TestPositiveValueValidation:
    """Config functions reject non-positive values where semantically required."""

    @patch.dict(os.environ, {"NEXUS_OUTPUT_LIMIT_BYTES": "-1"})
    def test_negative_output_limit_rejected(self):
        with pytest.raises(ConfigurationError):
            get_global_output_limit()

    @patch.dict(os.environ, {"NEXUS_OUTPUT_LIMIT_BYTES": "0"})
    def test_zero_output_limit_rejected(self):
        with pytest.raises(ConfigurationError):
            get_global_output_limit()

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_ATTEMPTS": "-1"})
    def test_negative_retry_max_attempts_rejected(self):
        with pytest.raises(ConfigurationError):
            get_retry_max_attempts()

    @patch.dict(os.environ, {"NEXUS_RETRY_MAX_ATTEMPTS": "0"})
    def test_zero_retry_max_attempts_rejected(self):
        with pytest.raises(ConfigurationError):
            get_retry_max_attempts()

    @patch.dict(os.environ, {"NEXUS_TIMEOUT_SECONDS": "-1"})
    def test_negative_timeout_rejected(self):
        with pytest.raises(ConfigurationError):
            get_global_timeout()

    @patch.dict(os.environ, {"NEXUS_TIMEOUT_SECONDS": "0"})
    def test_zero_timeout_rejected(self):
        with pytest.raises(ConfigurationError):
            get_global_timeout()


class TestGetCLIDetectionTimeout:
    """Test get_cli_detection_timeout() function."""

    def test_cli_detection_timeout_default(self, monkeypatch):
        """No env var → returns 30 (default)."""
        monkeypatch.delenv("NEXUS_CLI_DETECTION_TIMEOUT", raising=False)
        assert get_cli_detection_timeout() == 30

    @patch.dict(os.environ, {"NEXUS_CLI_DETECTION_TIMEOUT": "10"})
    def test_cli_detection_timeout_custom(self):
        """NEXUS_CLI_DETECTION_TIMEOUT=10 → returns 10."""
        assert get_cli_detection_timeout() == 10

    @patch.dict(os.environ, {"NEXUS_CLI_DETECTION_TIMEOUT": "abc"})
    def test_cli_detection_timeout_invalid(self):
        """NEXUS_CLI_DETECTION_TIMEOUT=abc → ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_cli_detection_timeout()
        assert exc_info.value.config_key == "NEXUS_CLI_DETECTION_TIMEOUT"

    @patch.dict(os.environ, {"NEXUS_CLI_DETECTION_TIMEOUT": "0"})
    def test_cli_detection_timeout_zero_rejected(self):
        """NEXUS_CLI_DETECTION_TIMEOUT=0 → ConfigurationError (must be positive)."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_cli_detection_timeout()
        assert exc_info.value.config_key == "NEXUS_CLI_DETECTION_TIMEOUT"
        assert "positive" in str(exc_info.value)

    @patch.dict(os.environ, {"NEXUS_CLI_DETECTION_TIMEOUT": "-5"})
    def test_cli_detection_timeout_negative_rejected(self):
        """NEXUS_CLI_DETECTION_TIMEOUT=-5 → ConfigurationError (must be positive)."""
        with pytest.raises(ConfigurationError) as exc_info:
            get_cli_detection_timeout()
        assert exc_info.value.config_key == "NEXUS_CLI_DETECTION_TIMEOUT"
        assert "positive" in str(exc_info.value)


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
