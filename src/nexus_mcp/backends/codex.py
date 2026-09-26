"""Codex backend: one `codex app-server` process per prompt call, via openai-codex."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable

import anyio
import openai_codex.generated.v2_all as wire
from fastmcp.exceptions import ToolError
from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, CodexError, Sandbox
from openai_codex.models import Notification

from nexus_mcp.types import BackendInfo, Profile, PromptRequest, PromptResult

__all__ = ["AsyncCodex", "info", "run"]

NAME = "codex"
_SANDBOXES: dict[Profile, Sandbox] = {
    "read_only": Sandbox.read_only,
    "workspace_write": Sandbox.workspace_write,
    "full_access": Sandbox.full_access,
}
_LIMITS = frozenset(
    {
        wire.CodexErrorInfoValue.usage_limit_exceeded,
        wire.CodexErrorInfoValue.rate_limit_exceeded,
        wire.CodexErrorInfoValue.server_overloaded,
        wire.CodexErrorInfoValue.session_budget_exceeded,
    }
)


def _ignore_session(_session_id: str) -> None:
    return None


def _close_sdk_pipes(client: AsyncCodex) -> None:
    """Close SDK-owned pipes after shutdown, including on startup failure.

    Remove when openai-codex closes its stdout/stderr pipes in CodexClient.close().
    """
    sync = client._client._sync
    original_close = sync.close

    def close_with_pipes() -> None:
        proc = sync._proc  # SDK clears this before its public close returns.
        try:
            original_close()
        finally:
            if proc is not None:
                # The SDK kills after a timed-out wait but does not reap that killed process.
                proc.wait(timeout=2)
                if proc.stdout is not None:
                    proc.stdout.close()
                if proc.stderr is not None:
                    proc.stderr.close()

    sync.close = close_with_pipes  # type: ignore[method-assign]  # Instance-local SDK fix.


@contextlib.asynccontextmanager
async def _session() -> AsyncIterator[AsyncCodex]:
    """Open AsyncCodex so a cancellation during startup still closes the app-server.

    AsyncCodex launches the process on a worker thread and cleans up only on `Exception`
    (SDK api.py:329-340), so a cancelled `async with AsyncCodex()` would leak it. Startup
    runs as its own task; on any exit we let it finish, then close. The cleanup is shielded
    because MCP cancels requests with an AnyIO cancel scope, which re-cancels every await.
    """
    client = AsyncCodex(CodexConfig(experimental_api=False))
    _close_sdk_pipes(client)
    startup = asyncio.ensure_future(client.__aenter__())
    try:
        await asyncio.shield(startup)
        yield client
    finally:
        with anyio.CancelScope(shield=True):
            if not startup.done():
                with contextlib.suppress(Exception):
                    await startup  # closing before the launch finishes would miss the process
            await client.close()


async def info() -> BackendInfo:
    """List visible models via app-server; still report installed if it cannot start."""
    try:
        async with _session() as client:
            listed = await client.models()
    except (CodexError, OSError):
        return BackendInfo(
            name=NAME, installed=True, hint="Could not start codex app-server to list models"
        )
    return BackendInfo(name=NAME, installed=True, models=[model.id for model in listed.data])


def _turn_error(turn: wire.Turn) -> ToolError:
    """Build an error from the enumerated error kind only; never copy provider prose."""
    if turn.status == wire.TurnStatus.interrupted:
        return ToolError("Codex turn was interrupted")
    detail = turn.error.codex_error_info if turn.error is not None else None
    kind = detail.root if detail is not None else None
    if isinstance(kind, wire.CodexErrorInfoValue):
        # Only enum kinds are hashable; object-shaped kinds (e.g. HttpConnectionFailed) are not.
        if kind == wire.CodexErrorInfoValue.unauthorized:
            return ToolError("Codex authentication failed; run `codex login`")
        if kind in _LIMITS:
            return ToolError(
                "Codex is rate limited, over its usage limit, or overloaded; retry later"
            )
        return ToolError(f"Codex turn failed (kind={kind.value})")
    if kind is not None:
        return ToolError(f"Codex turn failed (kind={type(kind).__name__})")
    return ToolError("Codex turn failed")


async def _collect(thread_id: str, events: AsyncIterator[Notification]) -> PromptResult:
    """Consume one turn's notifications; mirrors the SDK's final-answer selection rule."""
    final: str | None = None
    unphased: str | None = None
    usage: wire.ThreadTokenUsage | None = None
    turn: wire.Turn | None = None
    async for event in events:
        payload = event.payload
        if isinstance(payload, wire.ItemCompletedNotification):
            item = payload.item.root
            if isinstance(item, wire.AgentMessageThreadItem):
                if item.phase == wire.MessagePhase.final_answer:
                    final = item.text
                elif item.phase is None:
                    unphased = item.text
        elif isinstance(payload, wire.ThreadTokenUsageUpdatedNotification):
            usage = payload.token_usage
        elif isinstance(payload, wire.TurnCompletedNotification):
            turn = payload.turn
    if turn is None:
        raise ToolError("Codex stream ended without completing the turn")
    if turn.status != wire.TurnStatus.completed:
        raise _turn_error(turn)
    output = final if final is not None else unphased
    if output is None:
        raise ToolError("Codex finished the turn without a final answer")
    return PromptResult(
        backend=NAME,
        session_id=thread_id,
        output=output,
        usage=None if usage is None else usage.model_dump(mode="json"),
    )


async def run(
    req: PromptRequest, on_session: Callable[[str], None] = _ignore_session
) -> PromptResult:
    """Run one turn; new, continued (session_id), or forked (session_id + fork)."""
    sandbox = _SANDBOXES[req.profile]
    cwd = str(req.cwd)
    deny = ApprovalMode.deny_all
    try:
        async with _session() as client:
            if req.session_id is None:
                thread = await client.thread_start(
                    cwd=cwd, model=req.model, sandbox=sandbox, approval_mode=deny
                )
            elif req.fork:
                thread = await client.thread_fork(
                    req.session_id, cwd=cwd, model=req.model, sandbox=sandbox, approval_mode=deny
                )
            else:
                thread = await client.thread_resume(
                    req.session_id, cwd=cwd, model=req.model, sandbox=sandbox, approval_mode=deny
                )
            on_session(thread.id)
            # The profile is re-sent on the turn so a resumed/forked thread cannot keep a
            # wider sandbox than this call's profile.
            handle = await thread.turn(
                req.prompt, cwd=cwd, model=req.model, sandbox=sandbox, approval_mode=deny
            )
            try:
                return await _collect(thread.id, handle.stream())
            except asyncio.CancelledError:
                # AsyncCodex runs on worker threads; cancelling the task does not stop the turn.
                with anyio.CancelScope(shield=True):
                    with contextlib.suppress(CodexError, OSError):
                        await handle.interrupt()
                raise
    except FileNotFoundError:
        raise ToolError("Codex CLI binary not found; reinstall 'nexus-mcp[codex]'") from None
    except CodexError as error:
        raise ToolError(f"Codex run failed ({type(error).__name__})") from None
