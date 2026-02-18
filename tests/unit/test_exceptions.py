# tests/unit/test_exceptions.py
from nexus_mcp.exceptions import (
    ConfigurationError,
    NexusMCPError,
    ParseError,
    SubprocessError,
    SubprocessTimeoutError,
    UnsupportedAgentError,
)


def test_exception_hierarchy():
    assert issubclass(SubprocessError, NexusMCPError)
    assert issubclass(ParseError, NexusMCPError)
    assert issubclass(UnsupportedAgentError, NexusMCPError)


def test_subprocess_timeout_error_hierarchy():
    assert issubclass(SubprocessTimeoutError, SubprocessError)
    assert issubclass(SubprocessTimeoutError, NexusMCPError)


def test_subprocess_error_message():
    err = SubprocessError("Command failed", stderr="error output")
    assert "Command failed" in str(err)
    assert err.stderr == "error output"


def test_subprocess_error_stores_command_and_returncode():
    err = SubprocessError(
        "Command failed",
        stderr="error output",
        command=["gemini", "-p", "test"],
        returncode=1,
    )
    assert err.command == ["gemini", "-p", "test"]
    assert err.returncode == 1
    assert err.stderr == "error output"


def test_subprocess_error_defaults():
    err = SubprocessError("Command failed")
    assert err.stderr == ""
    assert err.command is None
    assert err.returncode is None


def test_subprocess_timeout_error_stores_timeout():
    err = SubprocessTimeoutError("Timed out", timeout=30.0)
    assert err.timeout == 30.0
    assert "Timed out" in str(err)


def test_subprocess_timeout_error_stores_command():
    err = SubprocessTimeoutError(
        "Timed out",
        timeout=30.0,
        command=["slow-command", "--flag"],
    )
    assert err.timeout == 30.0
    assert err.command == ["slow-command", "--flag"]


def test_parse_error_stores_raw_output():
    err = ParseError("Failed to parse JSON", raw_output='{"truncated": "da')
    assert "Failed to parse JSON" in str(err)
    assert err.raw_output == '{"truncated": "da'


def test_parse_error_defaults():
    err = ParseError("Parse failed")
    assert err.raw_output == ""


def test_unsupported_agent_error_stores_agent():
    err = UnsupportedAgentError("unknown-agent")
    assert err.agent == "unknown-agent"
    assert "Unsupported agent: unknown-agent" in str(err)


def test_configuration_error_stores_config_key():
    """ConfigurationError should store config_key attribute."""
    err = ConfigurationError("Invalid timeout value", config_key="NEXUS_TIMEOUT_SECONDS")
    assert str(err) == "Invalid timeout value"
    assert err.config_key == "NEXUS_TIMEOUT_SECONDS"


def test_configuration_error_default_config_key():
    """ConfigurationError config_key defaults to None."""
    err = ConfigurationError("Bad config")
    assert err.config_key is None


def test_subprocess_error_accepts_stdout():
    """SubprocessError should store stdout attribute."""
    err = SubprocessError("Command failed", stdout="output data")
    assert err.stdout == "output data"


def test_subprocess_error_stdout_defaults_to_empty():
    """SubprocessError stdout defaults to empty string."""
    err = SubprocessError("Command failed")
    assert err.stdout == ""


def test_subprocess_error_str_includes_returncode():
    """str(SubprocessError) should include returncode when set."""
    err = SubprocessError("Command failed", returncode=1)
    assert "returncode=1" in str(err)


def test_subprocess_error_str_includes_stderr():
    """str(SubprocessError) should include stderr when non-empty."""
    err = SubprocessError("Command failed", stderr="some error")
    assert "stderr='some error'" in str(err)


def test_subprocess_error_str_includes_stdout():
    """str(SubprocessError) should include stdout when non-empty."""
    err = SubprocessError("Command failed", stdout="some output")
    assert "stdout='some output'" in str(err)


def test_subprocess_error_str_omits_empty_fields():
    """str(SubprocessError) should not include labels for empty/None fields."""
    err = SubprocessError("Command failed")
    result = str(err)
    assert "returncode" not in result
    assert "stderr" not in result
    assert "stdout" not in result
    assert result == "Command failed"


def test_subprocess_error_str_truncates_long_stderr():
    """str(SubprocessError) truncates stderr longer than 500 chars to prevent data leakage."""
    long_stderr = "x" * 600
    err = SubprocessError("Command failed", stderr=long_stderr)
    result = str(err)
    assert "[truncated]" in result
    assert "x" * 501 not in result  # truncated before 501 chars


def test_subprocess_error_str_truncates_long_stdout():
    """str(SubprocessError) truncates stdout longer than 500 chars to prevent data leakage."""
    long_stdout = "y" * 600
    err = SubprocessError("Command failed", stdout=long_stdout)
    result = str(err)
    assert "[truncated]" in result
    assert "y" * 501 not in result


def test_subprocess_error_str_does_not_truncate_short_output():
    """str(SubprocessError) does not truncate stderr/stdout at or below 500 chars."""
    short_stderr = "short error"
    err = SubprocessError("Command failed", stderr=short_stderr)
    result = str(err)
    assert "[truncated]" not in result
    assert short_stderr in result


def test_subprocess_error_str_preserves_full_output_in_attributes():
    """Truncation in __str__ does not affect the stored stderr/stdout attributes."""
    long_stderr = "e" * 600
    err = SubprocessError("Command failed", stderr=long_stderr)
    assert len(err.stderr) == 600  # attribute is unmodified


def test_subprocess_timeout_error_accepts_stdout():
    """SubprocessTimeoutError should accept and store stdout parameter."""
    err = SubprocessTimeoutError("Timed out", timeout=30.0, stdout="partial output")
    assert err.stdout == "partial output"


def test_subprocess_timeout_error_stdout_defaults_to_empty():
    """SubprocessTimeoutError stdout defaults to empty string."""
    err = SubprocessTimeoutError("Timed out", timeout=30.0)
    assert err.stdout == ""
