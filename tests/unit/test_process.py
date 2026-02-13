# tests/unit/test_process.py
from unittest.mock import patch

import pytest

from nexus_mcp.exceptions import SubprocessError, SubprocessTimeoutError
from nexus_mcp.process import run_subprocess
from tests.fixtures import create_mock_process


# IMPORTANT: This patch target requires process.py to use "import asyncio" at module level
# See Step 7 header warning for details
@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_success(mock_exec):
    mock_exec.return_value = create_mock_process(stdout="output", stderr="", returncode=0)
    result = await run_subprocess(["echo", "hello"])
    assert result.stdout == "output"
    assert result.returncode == 0


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_captures_stderr(mock_exec):
    mock_exec.return_value = create_mock_process(stdout="", stderr="error", returncode=1)
    result = await run_subprocess(["false"])
    assert result.stderr == "error"
    assert result.returncode == 1


# Error scenario tests
@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_handles_unicode_errors(mock_exec):
    """Handle non-UTF-8 output gracefully."""
    mock_exec.return_value = create_mock_process(
        stdout_bytes=b"\xff\xfe",  # Invalid UTF-8
        returncode=0,
    )

    with pytest.raises(SubprocessError, match="decode"):
        await run_subprocess(["binary-cmd"])


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_handles_partial_output(mock_exec):
    """Handle subprocess killed mid-output (partial JSON)."""
    mock_exec.return_value = create_mock_process(
        stdout='{"response": "incomp',  # Truncated JSON
        stderr="",
        returncode=-9,  # SIGKILL
    )
    result = await run_subprocess(["killed-cmd"])
    assert result.returncode == -9
    assert "incomp" in result.stdout


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_cli_not_found(mock_exec):
    """Handle CLI binary not found in PATH."""
    mock_exec.side_effect = FileNotFoundError("gemini: command not found")

    with pytest.raises(SubprocessError, match="not found"):
        await run_subprocess(["gemini", "-p", "test"])


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_permission_denied(mock_exec):
    """Handle permission denied errors."""
    mock_exec.side_effect = PermissionError("Permission denied")

    with pytest.raises(SubprocessError, match="[Pp]ermission"):
        await run_subprocess(["/protected/binary"])


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_timeout(mock_exec):
    """Handle subprocess timeout."""
    # Create a mock process that never completes
    mock_process = create_mock_process(stdout="", delay=10)
    mock_exec.return_value = mock_process

    with pytest.raises(SubprocessTimeoutError, match="timed out") as exc_info:
        await run_subprocess(["slow-command"], timeout=0.1)
    assert exc_info.value.command == ["slow-command"]


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_no_timeout(mock_exec):
    """Handle timeout=None (no timeout)."""
    mock_exec.return_value = create_mock_process(stdout="output", stderr="", returncode=0)
    result = await run_subprocess(["echo", "hello"], timeout=None)
    assert result.stdout == "output"
    assert result.returncode == 0


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_error_includes_command(mock_exec):
    """Verify SubprocessError stores the failed command."""
    mock_exec.side_effect = FileNotFoundError("not found")

    with pytest.raises(SubprocessError) as exc_info:
        await run_subprocess(["gemini", "-p", "test"])
    assert exc_info.value.command == ["gemini", "-p", "test"]
