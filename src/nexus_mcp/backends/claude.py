"""Claude Agent SDK backend: one Claude Code CLI process per prompt call."""

from collections.abc import Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLINotFoundError,
    ResultError,
    ResultMessage,
    SystemMessage,
)
from fastmcp.exceptions import ToolError

from nexus_mcp.backends.claude_policy import build_options
from nexus_mcp.types import BackendInfo, PromptRequest, PromptResult

__all__ = ["ClaudeSDKClient", "info", "run"]

NAME = "claude"
_KNOWN_SUBTYPES = frozenset(
    {
        "success",
        "error_during_execution",
        "error_max_budget_usd",
        "error_max_structured_output_retries",
        "error_max_turns",
    }
)


async def info() -> BackendInfo:
    """Claude has no model listing; `model` accepts any id or alias the CLI accepts."""
    return BackendInfo(name=NAME, installed=True, models=None)


def _classify(subtype: str | None, status: int | None, assistant_error: str | None) -> ToolError:
    """Build an error from fixed text and safe fields only; never copy provider prose."""
    if assistant_error == "authentication_failed" or status == 401:
        return ToolError(
            "Claude authentication failed; run `claude` to log in or set ANTHROPIC_API_KEY"
        )
    if assistant_error == "rate_limit" or status == 429 or (status and 500 <= status < 600):
        return ToolError("Claude is rate limited or overloaded; retry later")
    if assistant_error == "billing_error":
        return ToolError("Claude reported a billing error")
    safe = subtype if subtype in _KNOWN_SUBTYPES else "unknown"
    return ToolError(f"Claude ended the turn with an error (subtype={safe})")


def _ignore_session(_session_id: str) -> None:
    return None


def _session_of(message: object) -> str | None:
    """Session ID carried by a message: SystemMessage init data, else session_id."""
    if isinstance(message, SystemMessage):
        value = message.data.get("session_id")
    else:
        value = getattr(message, "session_id", None)
    return value if isinstance(value, str) and value else None


async def run(
    req: PromptRequest, on_session: Callable[[str], None] = _ignore_session
) -> PromptResult:
    """Run one turn; new, continued (session_id), or forked (session_id + fork)."""
    options = build_options(
        cwd=req.cwd, profile=req.profile, model=req.model, resume=req.session_id, fork=req.fork
    )
    final: ResultMessage | None = None
    assistant_error: str | None = None
    announced = False
    try:
        async with ClaudeSDKClient(options) as client:
            await client.query(req.prompt)
            async for message in client.receive_response():
                session_id = _session_of(message)
                if not announced and session_id is not None:
                    announced = True
                    on_session(session_id)
                if isinstance(message, AssistantMessage) and message.error is not None:
                    assistant_error = message.error
                elif isinstance(message, ResultMessage):
                    final = message
    except CLINotFoundError:
        raise ToolError(
            "Claude Code CLI not found; install `claude` on PATH (required on Windows)"
        ) from None
    except ResultError as error:
        raise _classify(error.subtype, error.api_error_status, assistant_error) from None
    except ClaudeSDKError as error:
        raise ToolError(f"Claude run failed ({type(error).__name__})") from None
    if final is None:
        raise ToolError("Claude ended without a result")
    if final.is_error:
        raise _classify(final.subtype, final.api_error_status, assistant_error)
    return PromptResult(
        backend=NAME, session_id=final.session_id, output=final.result or "", usage=final.usage
    )
