# tests/unit/test_process.py
import asyncio
import os
import signal
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nexus_mcp.exceptions import (
    SubprocessError,
    SubprocessTimeoutError,
    SubprocessTreeTerminationError,
)
from nexus_mcp.process import run_subprocess
from tests.fixtures import REPRESENTATIVE_CLI, create_mock_process


# IMPORTANT: This patch target requires process.py to use "import asyncio" at module level
# See Step 7 header warning for details
@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_success(mock_exec):
    mock_exec.return_value = create_mock_process(stdout="output", stderr="", returncode=0)
    result = await run_subprocess(["echo", "hello"])
    assert result.stdout == "output"
    assert result.returncode == 0


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_forwards_non_none_cwd_only(mock_exec, tmp_path: Path):
    """Explicit workspaces reach subprocess creation without changing the default call shape."""
    mock_exec.return_value = create_mock_process(stdout="output", returncode=0)

    await run_subprocess(["echo", "hello"])
    call = mock_exec.await_args
    assert call.args == ("echo", "hello")
    assert call.kwargs["stdin"] is asyncio.subprocess.DEVNULL
    assert call.kwargs["stdout"] is asyncio.subprocess.PIPE
    assert call.kwargs["stderr"] is asyncio.subprocess.PIPE
    if os.name == "posix":
        assert call.kwargs["start_new_session"] is True
    else:
        assert int(call.kwargs["creationflags"]) != 0
    assert "cwd" not in call.kwargs

    mock_exec.reset_mock()
    workspace_path = tmp_path / "workspace"
    await run_subprocess(["pwd"], cwd=workspace_path)
    assert mock_exec.await_args.args == ("pwd",)
    assert mock_exec.await_args.kwargs["cwd"] == workspace_path


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
async def test_unicode_error_preserves_decodable_stdout(mock_exec):
    """When stderr has invalid UTF-8 but stdout is valid, stdout is preserved in the error."""
    mock_exec.return_value = create_mock_process(
        stdout_bytes=b"valid stdout",
        stderr_bytes=b"\xff\xfe",  # Invalid UTF-8
        returncode=0,
    )
    with pytest.raises(SubprocessError) as exc_info:
        await run_subprocess(["cmd"])
    assert exc_info.value.stdout == "valid stdout"


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
    mock_exec.side_effect = FileNotFoundError(f"{REPRESENTATIVE_CLI}: command not found")

    with pytest.raises(SubprocessError, match="not found"):
        await run_subprocess([REPRESENTATIVE_CLI, "-p", "test"])


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_permission_denied(mock_exec):
    """Handle permission denied errors."""
    mock_exec.side_effect = PermissionError("Permission denied")

    with pytest.raises(SubprocessError, match="[Pp]ermission"):
        await run_subprocess(["/protected/binary"])


@patch("nexus_mcp.process._terminate_process_tree", new_callable=AsyncMock, return_value=True)
@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_run_subprocess_timeout(mock_exec, _mock_terminate_process_tree):
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
        await run_subprocess([REPRESENTATIVE_CLI, "-p", "test"])
    assert exc_info.value.command == [REPRESENTATIVE_CLI, "-p", "test"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
@patch("nexus_mcp.process.os.killpg")
@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_timeout_kills_and_waits_for_owned_process_group(mock_exec, mock_killpg):
    """A timeout signals the isolated POSIX group and reaps its direct child."""
    mock_process = create_mock_process(stdout="", delay=10)
    mock_exec.return_value = mock_process

    with pytest.raises(SubprocessTimeoutError):
        await run_subprocess(["slow-command"], timeout=0.01)

    mock_killpg.assert_called_once_with(mock_process.pid, signal.SIGKILL)
    mock_process.kill.assert_not_called()
    mock_process.wait.assert_awaited_once()


@pytest.mark.parametrize("returncode", [None, 0])
@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
@patch("nexus_mcp.process.os.killpg")
@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_external_cancellation_reaps_subprocess(
    mock_exec,
    mock_killpg,
    returncode: int | None,
):
    """Caller cancellation cannot orphan a running child and still propagates cancellation."""
    mock_process = create_mock_process(stdout="", delay=10)
    mock_process.returncode = returncode
    mock_exec.return_value = mock_process
    task = asyncio.create_task(run_subprocess(["slow-command"], timeout=None))
    await asyncio.sleep(0)
    assert mock_process.communicate.await_count == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    mock_killpg.assert_called_once_with(mock_process.pid, signal.SIGKILL)
    mock_process.kill.assert_not_called()
    mock_process.wait.assert_awaited_once()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
@patch("nexus_mcp.process.os.killpg", side_effect=PermissionError("not owned"))
@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_unproven_tree_stoppage_raises_recoverable_process_error(mock_exec, _mock_killpg):
    """Direct-child cleanup cannot masquerade as proof that every descendant stopped."""
    mock_process = create_mock_process(stdout="", delay=10)
    mock_process.returncode = None
    mock_exec.return_value = mock_process
    task = asyncio.create_task(run_subprocess(["slow-command"], timeout=None))
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(SubprocessTreeTerminationError):
        await task

    mock_process.kill.assert_called_once()
    mock_process.wait.assert_awaited_once()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group falsifier")
async def test_timeout_stops_real_grandchild_process_tree(tmp_path: Path):
    """A child-spawned grandchild cannot mutate the workspace after timeout returns."""
    marker = tmp_path / "grandchild-finished"
    script = tmp_path / "spawn_tree.py"
    grandchild = (
        "import pathlib,sys,time; time.sleep(0.3); pathlib.Path(sys.argv[1]).write_text('late')"
    )
    script.write_text(
        "\n".join(
            [
                "import subprocess",
                "import sys",
                "import time",
                "marker = sys.argv[1]",
                f"subprocess.Popen([sys.executable, '-c', {grandchild!r}, marker])",
                "time.sleep(30)",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SubprocessTimeoutError):
        await run_subprocess([sys.executable, str(script), str(marker)], timeout=0.05)
    await asyncio.sleep(0.4)

    assert marker.exists() is False


@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_success_with_nonempty_stderr(mock_exec):
    """returncode=0 with stderr content → result.stderr populated, returncode=0."""
    mock_exec.return_value = create_mock_process(stdout="output", stderr="warning", returncode=0)

    result = await run_subprocess(["some-command"])

    assert result.stderr == "warning"
    assert result.returncode == 0
