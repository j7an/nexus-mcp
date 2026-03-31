"""Tests for configuration module (environment variable settings)."""

import logging
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from nexus_mcp.config import (
    HARDCODED_DEFAULTS,
    _merge_defaults,
    _read_global_env_defaults,
    _read_runner_env_defaults,
    get_agent_env,
    get_cli_detection_timeout,
    get_global_output_limit,
    get_global_timeout,
    get_retry_base_delay,
    get_retry_max_attempts,
    get_retry_max_delay,
    get_runner_defaults,
    get_runner_models,
    get_tool_timeout,
)
from nexus_mcp.exceptions import ConfigurationError
from nexus_mcp.types import OperationalDefaults


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


# ---------------------------------------------------------------------------
# Operational defaults model, hardcoded defaults, merge logic
# ---------------------------------------------------------------------------


class TestOperationalDefaults:
    """Test OperationalDefaults Pydantic model validation."""

    def test_all_none_is_valid(self):
        od = OperationalDefaults()
        assert od.timeout is None
        assert od.retry_base_delay is None
        assert od.execution_mode is None

    def test_ge1_fields_reject_zero(self):
        """timeout, output_limit, max_retries, cli_detection_timeout require ge=1."""
        for field in ("timeout", "output_limit", "max_retries", "cli_detection_timeout"):
            with pytest.raises(ValidationError):
                OperationalDefaults(**{field: 0})  # type: ignore[arg-type]

    def test_ge0_fields_allow_zero(self):
        """retry_base_delay, retry_max_delay, tool_timeout allow 0.0."""
        od = OperationalDefaults(retry_base_delay=0.0, retry_max_delay=0.0, tool_timeout=0.0)
        assert od.retry_base_delay == 0.0
        assert od.retry_max_delay == 0.0
        assert od.tool_timeout == 0.0

    def test_ge0_fields_reject_negative(self):
        for field in ("retry_base_delay", "retry_max_delay", "tool_timeout"):
            with pytest.raises(ValidationError):
                OperationalDefaults(**{field: -0.1})  # type: ignore[arg-type]

    def test_float_fields_reject_inf(self):
        with pytest.raises(ValidationError, match="must be a finite number"):
            OperationalDefaults(retry_base_delay=float("inf"))

    def test_float_fields_reject_nan(self):
        # ge=0 catches nan before the custom validator (nan >= 0 is False in Python),
        # so the error message is Pydantic's built-in ge constraint message, not ours.
        with pytest.raises(ValidationError):
            OperationalDefaults(retry_max_delay=float("nan"))

    def test_frozen(self):
        od = OperationalDefaults(timeout=10)
        with pytest.raises(ValidationError):
            od.timeout = 20  # type: ignore[misc]


class TestHardcodedDefaults:
    """HARDCODED_DEFAULTS provides expected baseline values."""

    def test_timeout(self):
        assert HARDCODED_DEFAULTS.timeout == 600

    def test_output_limit(self):
        assert HARDCODED_DEFAULTS.output_limit == 50000

    def test_max_retries(self):
        assert HARDCODED_DEFAULTS.max_retries == 3

    def test_retry_base_delay(self):
        assert HARDCODED_DEFAULTS.retry_base_delay == 2.0

    def test_retry_max_delay(self):
        assert HARDCODED_DEFAULTS.retry_max_delay == 60.0

    def test_tool_timeout(self):
        assert HARDCODED_DEFAULTS.tool_timeout == 900.0

    def test_cli_detection_timeout(self):
        assert HARDCODED_DEFAULTS.cli_detection_timeout == 30

    def test_execution_mode(self):
        assert HARDCODED_DEFAULTS.execution_mode == "default"


class TestMergeDefaults:
    """Test _merge_defaults overlay logic."""

    def test_base_preserved_when_overlay_is_none(self):
        base = OperationalDefaults(timeout=600)
        result = _merge_defaults(base, OperationalDefaults())
        assert result.timeout == 600

    def test_overlay_wins_over_base(self):
        base = OperationalDefaults(timeout=600)
        result = _merge_defaults(base, OperationalDefaults(timeout=300))
        assert result.timeout == 300

    def test_none_overlay_does_not_clear_base(self):
        base = OperationalDefaults(timeout=600)
        result = _merge_defaults(base, OperationalDefaults(timeout=None))
        assert result.timeout == 600

    def test_multiple_overlays_last_wins(self):
        base = OperationalDefaults(timeout=600)
        result = _merge_defaults(
            base, OperationalDefaults(timeout=300), OperationalDefaults(timeout=100)
        )
        assert result.timeout == 100

    def test_zero_float_wins_over_base(self):
        """0.0 is a valid non-None value — not filtered by truthiness."""
        base = OperationalDefaults(retry_base_delay=2.0)
        result = _merge_defaults(base, OperationalDefaults(retry_base_delay=0.0))
        assert result.retry_base_delay == 0.0


class TestReadGlobalEnvDefaults:
    """Test _read_global_env_defaults env var parsing (set vs unset distinction)."""

    def test_no_env_vars_returns_all_none(self, monkeypatch):
        for var in (
            "NEXUS_TIMEOUT_SECONDS",
            "NEXUS_OUTPUT_LIMIT_BYTES",
            "NEXUS_RETRY_MAX_ATTEMPTS",
            "NEXUS_CLI_DETECTION_TIMEOUT",
            "NEXUS_RETRY_BASE_DELAY",
            "NEXUS_RETRY_MAX_DELAY",
            "NEXUS_TOOL_TIMEOUT_SECONDS",
            "NEXUS_EXECUTION_MODE",
        ):
            monkeypatch.delenv(var, raising=False)
        result = _read_global_env_defaults()
        assert result.timeout is None
        assert result.retry_base_delay is None
        assert result.max_retries is None
        assert result.execution_mode is None

    def test_only_set_vars_are_populated(self, monkeypatch):
        monkeypatch.setenv("NEXUS_TIMEOUT_SECONDS", "300")
        monkeypatch.delenv("NEXUS_OUTPUT_LIMIT_BYTES", raising=False)
        result = _read_global_env_defaults()
        assert result.timeout == 300
        assert result.output_limit is None  # not set → stays None

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "0.0"})
    def test_zero_float_populated_not_filtered(self):
        """0.0 is valid — not treated as falsy."""
        result = _read_global_env_defaults()
        assert result.retry_base_delay == 0.0

    @patch.dict(os.environ, {"NEXUS_TIMEOUT_SECONDS": "bad"})
    def test_invalid_int_raises(self):
        with pytest.raises(ConfigurationError) as exc_info:
            _read_global_env_defaults()
        assert exc_info.value.config_key == "NEXUS_TIMEOUT_SECONDS"

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "nan"})
    def test_nan_raises_finite_error(self):
        with pytest.raises(ConfigurationError, match="must be a finite number"):
            _read_global_env_defaults()

    @patch.dict(os.environ, {"NEXUS_EXECUTION_MODE": "yolo"})
    def test_execution_mode_yolo(self):
        result = _read_global_env_defaults()
        assert result.execution_mode == "yolo"

    @patch.dict(os.environ, {"NEXUS_EXECUTION_MODE": "default"})
    def test_execution_mode_default(self):
        result = _read_global_env_defaults()
        assert result.execution_mode == "default"

    @patch.dict(os.environ, {"NEXUS_EXECUTION_MODE": "invalid"})
    def test_execution_mode_invalid_raises(self):
        with pytest.raises(ConfigurationError) as exc_info:
            _read_global_env_defaults()
        assert exc_info.value.config_key == "NEXUS_EXECUTION_MODE"
        assert "must be 'default' or 'yolo'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Per-runner env var defaults
# ---------------------------------------------------------------------------


class TestReadRunnerEnvDefaults:
    """Test _read_runner_env_defaults per-runner env var parsing."""

    def test_no_env_vars_returns_all_none(self, monkeypatch):
        for key in ("TIMEOUT", "OUTPUT_LIMIT", "MAX_RETRIES", "MODEL", "EXECUTION_MODE"):
            monkeypatch.delenv(f"NEXUS_GEMINI_{key}", raising=False)
        for key in ("RETRY_BASE_DELAY", "RETRY_MAX_DELAY"):
            monkeypatch.delenv(f"NEXUS_GEMINI_{key}", raising=False)
        result = _read_runner_env_defaults("gemini")
        assert result.timeout is None
        assert result.model is None

    def test_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_TIMEOUT", "900")
        result = _read_runner_env_defaults("gemini")
        assert result.timeout == 900

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_MODEL", "gemini-2.5-flash")
        result = _read_runner_env_defaults("gemini")
        assert result.model == "gemini-2.5-flash"

    def test_execution_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_EXECUTION_MODE", "yolo")
        result = _read_runner_env_defaults("gemini")
        assert result.execution_mode == "yolo"

    def test_invalid_execution_mode_silently_skipped(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_EXECUTION_MODE", "turbo")
        result = _read_runner_env_defaults("gemini")
        assert result.execution_mode is None  # silently skipped

    def test_retry_base_delay_from_env(self, monkeypatch):
        monkeypatch.setenv("NEXUS_CODEX_RETRY_BASE_DELAY", "0.5")
        result = _read_runner_env_defaults("codex")
        assert result.retry_base_delay == 0.5

    def test_zero_timeout_silently_skipped(self, monkeypatch):
        """NEXUS_GEMINI_TIMEOUT=0 must not produce a zero-second timeout."""
        monkeypatch.setenv("NEXUS_GEMINI_TIMEOUT", "0")
        result = _read_runner_env_defaults("gemini")
        assert result.timeout is None  # zero → skipped

    def test_negative_timeout_logs_warning(self, monkeypatch, caplog):
        """Negative per-runner integer env var logs a warning."""
        monkeypatch.setenv("NEXUS_GEMINI_TIMEOUT", "-5")
        with caplog.at_level(logging.WARNING, logger="nexus_mcp.config_resolver"):
            result = _read_runner_env_defaults("gemini")
        assert result.timeout is None
        assert "NEXUS_GEMINI_TIMEOUT" in caplog.text

    def test_negative_max_retries_silently_skipped(self, monkeypatch):
        """NEXUS_GEMINI_MAX_RETRIES=-5 must not produce range(-5)."""
        monkeypatch.setenv("NEXUS_GEMINI_MAX_RETRIES", "-5")
        result = _read_runner_env_defaults("gemini")
        assert result.max_retries is None

    def test_zero_max_retries_silently_skipped(self, monkeypatch):
        """NEXUS_GEMINI_MAX_RETRIES=0 must not produce range(0)."""
        monkeypatch.setenv("NEXUS_GEMINI_MAX_RETRIES", "0")
        result = _read_runner_env_defaults("gemini")
        assert result.max_retries is None

    def test_zero_output_limit_silently_skipped(self, monkeypatch):
        """Zero output limit must be silently skipped."""
        monkeypatch.setenv("NEXUS_GEMINI_OUTPUT_LIMIT", "0")
        result = _read_runner_env_defaults("gemini")
        assert result.output_limit is None

    def test_invalid_timeout_logs_warning(self, monkeypatch, caplog):
        """Invalid per-runner integer env var logs a warning."""
        monkeypatch.setenv("NEXUS_GEMINI_TIMEOUT", "not-a-number")
        with caplog.at_level(logging.WARNING, logger="nexus_mcp.config_resolver"):
            result = _read_runner_env_defaults("gemini")
        assert result.timeout is None
        assert "NEXUS_GEMINI_TIMEOUT" in caplog.text
        assert "not-a-number" in caplog.text

    def test_invalid_retry_delay_logs_warning(self, monkeypatch, caplog):
        """Invalid per-runner float env var logs a warning."""
        monkeypatch.setenv("NEXUS_CODEX_RETRY_BASE_DELAY", "banana")
        with caplog.at_level(logging.WARNING, logger="nexus_mcp.config_resolver"):
            result = _read_runner_env_defaults("codex")
        assert result.retry_base_delay is None
        assert "NEXUS_CODEX_RETRY_BASE_DELAY" in caplog.text
        assert "banana" in caplog.text

    def test_non_positive_timeout_logs_warning(self, monkeypatch, caplog):
        """Non-positive per-runner integer env var logs a warning."""
        monkeypatch.setenv("NEXUS_GEMINI_TIMEOUT", "0")
        with caplog.at_level(logging.WARNING, logger="nexus_mcp.config_resolver"):
            result = _read_runner_env_defaults("gemini")
        assert result.timeout is None
        assert "NEXUS_GEMINI_TIMEOUT" in caplog.text

    def test_negative_retry_delay_logs_warning(self, monkeypatch, caplog):
        """Negative per-runner float env var logs a warning."""
        monkeypatch.setenv("NEXUS_CODEX_RETRY_BASE_DELAY", "-1.0")
        with caplog.at_level(logging.WARNING, logger="nexus_mcp.config_resolver"):
            result = _read_runner_env_defaults("codex")
        assert result.retry_base_delay is None
        assert "NEXUS_CODEX_RETRY_BASE_DELAY" in caplog.text


# ---------------------------------------------------------------------------
# Per-runner models
# ---------------------------------------------------------------------------


class TestGetRunnerModels:
    """Test get_runner_models() env var parsing."""

    def test_not_set_returns_empty(self, monkeypatch):
        monkeypatch.delenv("NEXUS_GEMINI_MODELS", raising=False)
        assert get_runner_models("gemini") == ()

    def test_single_model(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_MODELS", "gemini-2.5-flash")
        assert get_runner_models("gemini") == ("gemini-2.5-flash",)

    def test_multiple_models(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-pro")
        assert get_runner_models("gemini") == ("gemini-2.5-flash", "gemini-2.5-pro")

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_MODELS", " flash , pro ")
        assert get_runner_models("gemini") == ("flash", "pro")

    def test_empty_entries_filtered(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_MODELS", "flash,,pro,")
        assert get_runner_models("gemini") == ("flash", "pro")

    def test_empty_string_returns_empty(self, monkeypatch):
        monkeypatch.setenv("NEXUS_GEMINI_MODELS", "")
        assert get_runner_models("gemini") == ()


# ---------------------------------------------------------------------------
# Per-runner defaults (3-tier merge)
# ---------------------------------------------------------------------------


class TestGetRunnerDefaults:
    """Test get_runner_defaults() 3-tier merge: hardcoded → global env → per-runner env."""

    def test_unknown_runner_returns_hardcoded_defaults(self):
        result = get_runner_defaults("nonexistent")
        assert result.timeout == HARDCODED_DEFAULTS.timeout

    @patch.dict(os.environ, {"NEXUS_TIMEOUT_SECONDS": "300"})
    def test_global_env_overrides_hardcoded(self):
        result = get_runner_defaults("gemini")
        assert result.timeout == 300

    @patch.dict(os.environ, {"NEXUS_GEMINI_TIMEOUT": "900"})
    def test_runner_env_overrides_global(self):
        result = get_runner_defaults("gemini")
        assert result.timeout == 900

    @patch.dict(os.environ, {"NEXUS_TIMEOUT_SECONDS": "300", "NEXUS_GEMINI_TIMEOUT": "900"})
    def test_runner_env_wins_over_global_env(self):
        """Per-runner env has higher priority than global env."""
        result = get_runner_defaults("gemini")
        assert result.timeout == 900

    @patch.dict(os.environ, {"NEXUS_GEMINI_MODEL": "gemini-2.0-flash"})
    def test_runner_model_from_env(self):
        result = get_runner_defaults("gemini")
        assert result.model == "gemini-2.0-flash"

    def test_defaults_are_non_none_for_required_fields(self):
        """All required fields are non-None after merge with hardcoded defaults."""
        result = get_runner_defaults("gemini")
        assert result.timeout is not None
        assert result.output_limit is not None
        assert result.max_retries is not None
        assert result.retry_base_delay is not None
        assert result.retry_max_delay is not None


# ---------------------------------------------------------------------------
# Backward-compatible getter functions
# ---------------------------------------------------------------------------


class TestBackwardCompatGetters:
    """Backward-compat getters read env vars fresh (no singleton)."""

    def test_all_getters_return_hardcoded_defaults(self):
        assert get_global_timeout() == HARDCODED_DEFAULTS.timeout
        assert get_global_output_limit() == HARDCODED_DEFAULTS.output_limit
        assert get_retry_max_attempts() == HARDCODED_DEFAULTS.max_retries
        assert get_retry_base_delay() == HARDCODED_DEFAULTS.retry_base_delay
        assert get_retry_max_delay() == HARDCODED_DEFAULTS.retry_max_delay
        assert get_cli_detection_timeout() == HARDCODED_DEFAULTS.cli_detection_timeout

    def test_get_tool_timeout_zero_coercion(self, monkeypatch):
        monkeypatch.setenv("NEXUS_TOOL_TIMEOUT_SECONDS", "0")
        assert get_tool_timeout() is None

    @patch.dict(os.environ, {"NEXUS_TIMEOUT_SECONDS": "42"})
    def test_env_override_reaches_getter(self):
        assert get_global_timeout() == 42

    @patch.dict(os.environ, {"NEXUS_RETRY_BASE_DELAY": "0.0"})
    def test_zero_delay_preserved_by_getter(self):
        """0.0 is a valid base delay — getter must not coerce to default."""
        assert get_retry_base_delay() == 0.0
