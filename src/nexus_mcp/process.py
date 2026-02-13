# src/nexus_mcp/process.py
import asyncio  # ← REQUIRED: Module-level import for mock patching to work (see Step 7 warning)

from nexus_mcp.exceptions import SubprocessError, SubprocessTimeoutError
from nexus_mcp.types import SubprocessResult


async def run_subprocess(command: list[str], timeout: float | None = 600.0) -> SubprocessResult:
    """Execute a subprocess command and return result.

    Args:
        command: Command and arguments as list
        timeout: Maximum execution time in seconds (default: 600s / 10 minutes).
                 Pass None to disable timeout.

    Raises:
        SubprocessError: If command not found, permission denied, or decode errors.
        SubprocessTimeoutError: If command exceeds timeout.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        # Wait for process with optional timeout
        try:
            if timeout is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except TimeoutError:
            # Kill process and capture its exit code
            process.kill()
            await process.wait()
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
