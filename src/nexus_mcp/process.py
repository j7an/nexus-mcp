# src/nexus_mcp/process.py
__all__ = ["run_subprocess"]

import asyncio  # ← REQUIRED: Module-level import for mock patching to work (see Step 7 warning)
import os
import signal
import subprocess as stdlib_subprocess
from pathlib import Path
from typing import Any

from nexus_mcp.exceptions import (
    SubprocessError,
    SubprocessTimeoutError,
    SubprocessTreeTerminationError,
)
from nexus_mcp.types import SubprocessResult


async def run_subprocess(
    command: list[str],
    timeout: float | None = 600.0,
    *,
    cwd: Path | None = None,
) -> SubprocessResult:
    """Execute a subprocess command and return result.

    Args:
        command: Command and arguments as list
        timeout: Maximum execution time in seconds (default: 600s / 10 minutes).
                 Pass None to disable timeout.
        cwd: Canonical working directory. Omitted from subprocess creation when unset.

    Raises:
        SubprocessError: If command not found, permission denied, or decode errors.
        SubprocessTimeoutError: If command exceeds timeout.
    """
    try:
        subprocess_kwargs: dict[str, Any] = {
            "stdin": asyncio.subprocess.DEVNULL,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            **_process_tree_creation_kwargs(),
        }
        if cwd is not None:
            subprocess_kwargs["cwd"] = cwd
        process = await asyncio.create_subprocess_exec(*command, **subprocess_kwargs)

        # Wait for process with optional timeout
        try:
            if timeout is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.CancelledError:
            stopped = await _terminate_process_tree(process)
            if not stopped:
                raise SubprocessTreeTerminationError(
                    f"Could not prove subprocess tree stopped: {command[0]}",
                    command=command,
                    returncode=process.returncode,
                ) from None
            raise
        except TimeoutError:
            stopped = await _terminate_process_tree(process)
            if not stopped:
                raise SubprocessTreeTerminationError(
                    f"Could not prove subprocess tree stopped after timeout: {command[0]}",
                    command=command,
                    returncode=process.returncode,
                ) from None
            raise SubprocessTimeoutError(
                f"Command timed out after {timeout}s: {command[0]}",
                timeout=timeout,  # type: ignore[arg-type]  # timeout is not None in TimeoutError path
                command=command,
                returncode=process.returncode,  # Available after kill()+wait()
            ) from None

        # Decode output, handle encoding errors
        try:
            stdout = stdout_bytes.decode("utf-8")
            stderr = stderr_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SubprocessError(
                f"Failed to decode subprocess output: {e}",
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                command=command,
            ) from None

        assert process.returncode is not None  # Guaranteed after communicate()
        return SubprocessResult(
            stdout=stdout,
            stderr=stderr,
            returncode=process.returncode,
        )

    except FileNotFoundError as e:
        raise SubprocessError(
            f"Command not found: {command[0]}", stderr=str(e), command=command
        ) from None
    except PermissionError as e:
        raise SubprocessError(
            f"Permission denied: {command[0]}", stderr=str(e), command=command
        ) from None


def _process_tree_creation_kwargs() -> dict[str, object]:
    """Isolate every child in an OS-owned tree boundary before it can spawn descendants."""
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        create_group = getattr(stdlib_subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": create_group}
    return {}


async def _terminate_process_tree(process: asyncio.subprocess.Process) -> bool:
    """Terminate and await an owned tree, returning whether full-tree stoppage is proven."""
    stopped = False
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            stopped = True
        except ProcessLookupError:
            stopped = True
        except (OSError, TypeError, ValueError):
            stopped = False
    elif os.name == "nt":
        stopped = await asyncio.to_thread(_terminate_windows_tree, process.pid)

    if not stopped and process.returncode is None:
        process.kill()
    await process.wait()
    return stopped


def _terminate_windows_tree(pid: int) -> bool:
    """Use the Windows tree-aware system terminator for the isolated process group."""
    try:
        completed = stdlib_subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdin=stdlib_subprocess.DEVNULL,
            stdout=stdlib_subprocess.DEVNULL,
            stderr=stdlib_subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, stdlib_subprocess.SubprocessError):
        return False
    return completed.returncode == 0
