"""Tests for nexus_mcp.exceptions — custom exception hierarchy."""

from nexus_mcp.exceptions import (
    AgentNotFoundError,
    CLIProcessError,
    CLITimeoutError,
    NexusMCPError,
    ParseError,
)

# --- NexusMCPError (base) ---


class TestNexusMCPError:
    def test_is_exception_subclass(self) -> None:
        exc = NexusMCPError("something went wrong")
        assert isinstance(exc, Exception)

    def test_message_via_str(self) -> None:
        exc = NexusMCPError("something went wrong")
        assert str(exc) == "something went wrong"


# --- CLIProcessError ---


class TestCLIProcessError:
    def test_inherits_from_base(self) -> None:
        exc = CLIProcessError(returncode=1, stderr="error output", command="gemini -p hi")
        assert isinstance(exc, NexusMCPError)
        assert isinstance(exc, Exception)

    def test_stores_attributes(self) -> None:
        exc = CLIProcessError(returncode=2, stderr="fatal error", command="codex exec foo")
        assert exc.returncode == 2
        assert exc.stderr == "fatal error"
        assert exc.command == "codex exec foo"

    def test_message_contains_context(self) -> None:
        exc = CLIProcessError(returncode=127, stderr="not found", command="gemini -p test")
        msg = str(exc)
        assert "127" in msg
        assert "not found" in msg
        assert "gemini -p test" in msg


# --- CLITimeoutError ---


class TestCLITimeoutError:
    def test_inherits_from_base(self) -> None:
        exc = CLITimeoutError(timeout=30.0, command="gemini -p slow")
        assert isinstance(exc, NexusMCPError)
        assert isinstance(exc, Exception)

    def test_stores_attributes(self) -> None:
        exc = CLITimeoutError(timeout=60.5, command="codex exec long")
        assert exc.timeout == 60.5
        assert exc.command == "codex exec long"

    def test_message_contains_context(self) -> None:
        exc = CLITimeoutError(timeout=30.0, command="gemini -p slow")
        msg = str(exc)
        assert "30.0" in msg
        assert "gemini -p slow" in msg


# --- ParseError ---


class TestParseError:
    def test_inherits_from_base(self) -> None:
        exc = ParseError(raw_output="not json", agent="gemini")
        assert isinstance(exc, NexusMCPError)
        assert isinstance(exc, Exception)

    def test_stores_attributes(self) -> None:
        exc = ParseError(raw_output="<html>bad</html>", agent="codex")
        assert exc.raw_output == "<html>bad</html>"
        assert exc.agent == "codex"

    def test_message_contains_context(self) -> None:
        exc = ParseError(raw_output="garbled output", agent="gemini")
        msg = str(exc)
        assert "gemini" in msg
        assert "garbled output" in msg


# --- AgentNotFoundError ---


class TestAgentNotFoundError:
    def test_inherits_from_base(self) -> None:
        exc = AgentNotFoundError(agent="unknown", available=["gemini", "codex"])
        assert isinstance(exc, NexusMCPError)
        assert isinstance(exc, Exception)

    def test_stores_attributes(self) -> None:
        exc = AgentNotFoundError(agent="gpt4", available=["gemini", "codex", "claude"])
        assert exc.agent == "gpt4"
        assert exc.available == ["gemini", "codex", "claude"]

    def test_message_contains_context(self) -> None:
        exc = AgentNotFoundError(agent="gpt4", available=["gemini", "codex"])
        msg = str(exc)
        assert "gpt4" in msg
        assert "gemini" in msg
        assert "codex" in msg
