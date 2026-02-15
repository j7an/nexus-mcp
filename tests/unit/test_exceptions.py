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
