# tests/fixtures.py
import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

from nexus_mcp.types import AgentResponse, PromptRequest, SubprocessResult

# ---------------------------------------------------------------------------
# Reusable test constants
# ---------------------------------------------------------------------------

GEMINI_JSON_RESPONSE = '{"response": "test output"}'
GEMINI_JSON_WITH_STATS = (
    '{"response": "Hello, world!", "stats": {"models": {"gemini-2.5-flash": 1}}}'
)

# Phase 6/7: Add CODEX_NDJSON_RESPONSE and CLAUDE_JSON_RESPONSE constants
# when implementing CodexRunner and ClaudeCodeRunner respectively.


# ---------------------------------------------------------------------------
# Mock subprocess helper
# ---------------------------------------------------------------------------

_MOCK_PID = 12345


def create_mock_process(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    delay: float = 0,
    *,
    stdout_bytes: bytes | None = None,
    stderr_bytes: bytes | None = None,
) -> AsyncMock:
    """Create a mock ``asyncio.subprocess.Process`` for testing.

    Simulates the object returned by ``asyncio.create_subprocess_exec()``.

    Args:
        stdout: Simulated stdout output (encoded to UTF-8).
        stderr: Simulated stderr output (encoded to UTF-8).
        returncode: Simulated exit code.
        delay: Seconds to delay ``communicate()`` (for timeout tests).
        stdout_bytes: Raw stdout bytes (overrides ``stdout`` when set).
                      Use for testing non-UTF-8 / binary output scenarios.
        stderr_bytes: Raw stderr bytes (overrides ``stderr`` when set).

    Usage::

        @patch("nexus_mcp.process.asyncio.create_subprocess_exec")
        async def test_something(mock_exec):
            mock_exec.return_value = create_mock_process(
                stdout='{"response": "ok"}',
                returncode=0,
            )
            result = await run_subprocess(["echo", "hello"])
            assert result.returncode == 0
    """
    mock = AsyncMock()

    out = stdout_bytes if stdout_bytes is not None else stdout.encode("utf-8")
    err = stderr_bytes if stderr_bytes is not None else stderr.encode("utf-8")

    if delay > 0:
        # Use side_effect to preserve mock introspection (.call_count,
        # .assert_awaited_once(), etc.) while adding delay behavior.
        # NOTE: The delay works with asyncio.wait_for() because asyncio.sleep()
        # is cancellation-safe — wait_for cancels this coroutine on timeout,
        # which interrupts the sleep and raises TimeoutError in the caller.
        # Real communicate() returns tuple[bytes | None, bytes | None] but we always
        # pipe stdout/stderr, so bytes is correct for our tests.
        async def delayed_communicate() -> tuple[bytes, bytes]:
            await asyncio.sleep(delay)
            return (out, err)

        mock.communicate.side_effect = delayed_communicate
    else:
        mock.communicate.return_value = (out, err)

    mock.returncode = returncode
    mock.pid = _MOCK_PID  # Prevents MagicMock leaking if code logs pid

    # Stream attributes — explicitly None because create_subprocess_exec
    # is called with stdout=PIPE, stderr=PIPE (streams accessed via communicate()).
    mock.stdin = None
    mock.stdout = None
    mock.stderr = None

    # Process.kill() / send_signal() / terminate() are synchronous.
    # Process.wait() is a coroutine (waits for termination via event loop).
    # See: https://docs.python.org/3.12/library/asyncio-subprocess.html
    mock.kill = Mock()
    mock.send_signal = Mock()
    mock.terminate = Mock()
    mock.wait = AsyncMock()

    return mock


def make_prompt_request(**overrides: Any) -> PromptRequest:
    """Create a PromptRequest with sensible defaults.

    Usage::

        req = make_prompt_request()                          # defaults
        req = make_prompt_request(execution_mode="yolo")     # override one field
        req = make_prompt_request(agent="codex", prompt="X") # override multiple
    """
    defaults: dict[str, Any] = {"agent": "gemini", "prompt": "Hello"}
    return PromptRequest(**(defaults | overrides))


def make_agent_response(**overrides: Any) -> AgentResponse:
    """Create an AgentResponse with sensible defaults.

    Usage::

        resp = make_agent_response()                              # defaults
        resp = make_agent_response(output="custom")               # override one field
        resp = make_agent_response(agent="codex", output="Done")  # override multiple
    """
    defaults: dict[str, Any] = {
        "agent": "gemini",
        "output": "test output",
        "raw_output": GEMINI_JSON_RESPONSE,
    }
    return AgentResponse(**(defaults | overrides))


def make_subprocess_result(**overrides: Any) -> SubprocessResult:
    """Create a SubprocessResult with sensible defaults.

    Usage::

        res = make_subprocess_result()                          # defaults
        res = make_subprocess_result(returncode=1)              # override one field
        res = make_subprocess_result(stderr="err", returncode=2)  # override multiple
    """
    defaults: dict[str, Any] = {
        "stdout": GEMINI_JSON_RESPONSE,
        "stderr": "",
        "returncode": 0,
    }
    return SubprocessResult(**(defaults | overrides))
