# tests/fixtures.py
import asyncio
import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from nexus_mcp.cli_detector import CLIInfo
from nexus_mcp.runners.factory import RunnerFactory
from nexus_mcp.types import AgentResponse, AgentTask, PromptRequest, SessionPreferences

# ---------------------------------------------------------------------------
# Reusable test constants
# ---------------------------------------------------------------------------


def gemini_json(output: str, stats: dict | None = None) -> str:
    """Build a Gemini CLI JSON response string."""
    data: dict = {"response": output}
    if stats is not None:
        data["stats"] = stats
    return json.dumps(data)


def gemini_error_json(code: int, message: str, status: str) -> str:
    """Build a Gemini API error JSON string."""
    return json.dumps({"error": {"code": code, "message": message, "status": status}})


GEMINI_JSON_RESPONSE = gemini_json("test output")
GEMINI_JSON_WITH_STATS = gemini_json("Hello, world!", stats={"models": {"gemini-2.5-flash": 1}})

# Realistic Node.js warning prefix emitted by Gemini CLI v0.29.0+ before JSON output.
# Used to test parse_output() fallback for noisy stdout.
GEMINI_NOISY_STDOUT = (
    "(node:87799) [DEP0040] DeprecationWarning: The `punycode` module is deprecated."
    " Please use a userland alternative instead.\n"
    "(Use `node --trace-deprecation ...` to show where the warning was created)\n"
    "Loaded cached credentials.\n"
    '{"response": "test output"}'
)

CODEX_NDJSON_RESPONSE = "\n".join(
    [
        '{"type": "thread.started", "thread_id": "t1"}',
        '{"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "pong"}}',
        '{"type": "turn.completed", "turn_id": "r1"}',
    ]
)


def claude_json(
    result_text: str,
    cost_usd: float = 0.005,
    duration_ms: int = 5000,
    num_turns: int = 1,
) -> str:
    """Build a Claude Code CLI JSON response string."""
    data = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": result_text}],
            },
            "cost_usd": round(cost_usd * 0.2, 4),
            "duration_ms": duration_ms // 2,
        },
        {
            "type": "result",
            "result": result_text,
            "session_id": "test-session-id",
            "cost_usd": cost_usd,
            "duration_ms": duration_ms,
            "num_turns": num_turns,
        },
    ]
    return json.dumps(data)


CLAUDE_JSON_RESPONSE = claude_json("test output")


def opencode_ndjson(output: str) -> str:
    """Build an OpenCode NDJSON response string using the real OpenCode schema.

    OpenCode run --format json emits:
        {"type": "step_start", ...}
        {"type": "text", "part": {"type": "text", "text": "<output>"}}
        {"type": "step_finish", ...}
    """
    import json as _json

    return "\n".join(
        [
            '{"type": "step_start", "sessionID": "ses_test"}',
            _json.dumps(
                {
                    "type": "text",
                    "sessionID": "ses_test",
                    "part": {"type": "text", "text": output},
                }
            ),
            '{"type": "step_finish", "sessionID": "ses_test"}',
        ]
    )


def opencode_json(output: str) -> str:
    """Build an OpenCode single JSON object response string."""
    return json.dumps({"message": output})


OPENCODE_NDJSON_RESPONSE = opencode_ndjson("test output")
OPENCODE_JSON_RESPONSE = opencode_json("test output")


def claude_error_json(code: int | str, message: str) -> str:
    """Build a Claude API error JSON string."""
    return json.dumps({"error": {"code": code, "message": message}})


def codex_error_json(code: int | str, message: str) -> str:
    """Build a Codex API error JSON string."""
    return json.dumps({"error": {"code": code, "message": message}})


def opencode_error_json(name: str, message: str, *, status_code: int | None = None) -> str:
    """Build an OpenCode NDJSON error event string.

    Real format: {"type":"error","error":{"name":"...","data":{"message":"...","statusCode":N}}}
    """
    data: dict[str, object] = {"message": message}
    if status_code is not None:
        data["statusCode"] = status_code
    return json.dumps({"type": "error", "error": {"name": name, "data": data}})


CLAUDE_NOISY_STDOUT = "Loading configuration...\nConnecting to API...\n" + claude_json(
    "test output"
)

OPENCODE_NOISY_STDOUT = (
    "(node:12345) DeprecationWarning: some warning\n"
    "Loading state...\n" + opencode_ndjson("test output")
)

# ---------------------------------------------------------------------------
# Integration test constants
# ---------------------------------------------------------------------------

PING_PROMPT = "Reply with exactly the word 'pong'"


# ---------------------------------------------------------------------------
# CLI detection mock helpers
# ---------------------------------------------------------------------------


@contextmanager
def cli_detection_mocks():
    """Context manager that mocks Gemini CLI detection for unit tests.

    Patches detect_cli() and get_cli_version() so tests don't require
    the real Gemini binary. Version "0.12.0" → supports_json=True,
    keeping existing command assertions valid.

    Usage in conftest.py::

        @pytest.fixture
        def mock_cli_detection():
            with cli_detection_mocks() as mock:
                yield mock
    """
    with (
        patch("nexus_mcp.runners.base.detect_cli") as mock_detect,
        patch("nexus_mcp.runners.base.get_cli_version", return_value="0.12.0"),
    ):
        mock_detect.return_value = CLIInfo(found=True, path="/usr/bin/gemini")
        yield mock_detect
    RunnerFactory.clear_cache()


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
    # is called with stdin=DEVNULL, stdout=PIPE, stderr=PIPE (streams accessed via communicate()).
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
        req = make_prompt_request(cli="codex", prompt="X") # override multiple
    """
    defaults: dict[str, Any] = {"cli": "gemini", "prompt": "Hello"}
    return PromptRequest(**(defaults | overrides))


def make_agent_response(**overrides: Any) -> AgentResponse:
    """Create an AgentResponse with sensible defaults.

    Usage::

        resp = make_agent_response()                              # defaults
        resp = make_agent_response(output="custom")               # override one field
        resp = make_agent_response(cli="codex", output="Done")  # override multiple
    """
    defaults: dict[str, Any] = {
        "cli": "gemini",
        "output": "test output",
        "raw_output": GEMINI_JSON_RESPONSE,
    }
    return AgentResponse(**(defaults | overrides))


def make_agent_task(**overrides: Any) -> AgentTask:
    """Create an AgentTask with sensible defaults.

    Usage::

        task = make_agent_task()                              # defaults
        task = make_agent_task(cli="codex")                 # override cli
        task = make_agent_task(prompt="Do X", label="my-task")
    """
    defaults: dict[str, Any] = {"cli": "gemini", "prompt": "Hello", "execution_mode": "default"}
    return AgentTask(**(defaults | overrides))


def make_session_preferences(**overrides: Any) -> SessionPreferences:
    """Create a SessionPreferences with sensible defaults.

    Defaults mirror the post-construction state of a fresh session: no mode or
    model pinned, so all callers fall through to per-call or runner defaults.

    Usage::

        prefs = make_session_preferences()                         # execution_mode=None, model=None
        prefs = make_session_preferences(execution_mode="yolo")    # override one field
        prefs = make_session_preferences(execution_mode="yolo", model="gemini-2.5-flash")
    """
    defaults: dict[str, Any] = {
        "execution_mode": None,
        "model": None,
        "max_retries": None,
        "output_limit": None,
        "timeout": None,
    }
    return SessionPreferences(**(defaults | overrides))
