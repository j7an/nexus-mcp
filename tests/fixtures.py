"""Shared test fixtures and mock helpers for nexus-mcp tests.

This module provides:
- JSON response constants for Gemini CLI testing
- Mock subprocess.Process factory for unit testing runners
"""

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, Mock

# Gemini CLI JSON response samples (v0.12.0+ format)
GEMINI_JSON_RESPONSE = '{"response": "Hello from Gemini!"}'

GEMINI_JSON_WITH_STATS = """{
  "response": "Analysis complete",
  "stats": {
    "input_tokens": 42,
    "output_tokens": 128,
    "model": "gemini-2.5-flash"
  }
}"""

# Mock process ID for subprocess testing
_MOCK_PID = 12345


def create_mock_process(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    delay: float = 0.0,
    stdout_bytes: bytes | None = None,
    stderr_bytes: bytes | None = None,
) -> Mock:
    """Create a mock asyncio.subprocess.Process for testing.

    Args:
        stdout: String output to return from stdout (encoded as UTF-8)
        stderr: String output to return from stderr (encoded as UTF-8)
        returncode: Process exit code
        delay: Seconds to sleep before communicate() returns (for timeout tests)
        stdout_bytes: Override stdout with raw bytes (for encoding error tests)
        stderr_bytes: Override stderr with raw bytes (for encoding error tests)

    Returns:
        Mock Process object with async communicate/wait and sync kill/terminate
    """
    # Use raw bytes if provided, otherwise encode strings
    stdout_data = stdout_bytes if stdout_bytes is not None else stdout.encode("utf-8")
    stderr_data = stderr_bytes if stderr_bytes is not None else stderr.encode("utf-8")

    # Create async mock for communicate
    async def mock_communicate() -> tuple[bytes, bytes]:
        if delay > 0:
            await asyncio.sleep(delay)
        return (stdout_data, stderr_data)

    # Create async mock for wait
    async def mock_wait() -> int:
        if delay > 0:
            await asyncio.sleep(delay)
        return returncode

    # Build mock process
    process = Mock()
    process.pid = _MOCK_PID
    process.returncode = returncode
    process.stdout = BytesIO(stdout_data)
    process.stderr = BytesIO(stderr_data)

    # Async methods
    process.communicate = AsyncMock(side_effect=mock_communicate)
    process.wait = AsyncMock(side_effect=mock_wait)

    # Sync methods (for cleanup/termination)
    process.kill = Mock()
    process.terminate = Mock()
    process.send_signal = Mock()

    return process
