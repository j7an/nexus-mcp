"""Compound tools that chain multiple OpenCode HTTP calls.

Each tool aggregates data from multiple API endpoints and optionally
uses ctx.sample() for AI-powered summarization. When sampling is not
supported by the client, raw aggregated data is returned.

Tools:
- opencode_investigate: Search + read files + optional analysis
- opencode_session_review: Session + messages + diff + optional summary
"""

import logging
import re
from typing import Any

from fastmcp import Context, FastMCP

logger = logging.getLogger(__name__)


def _format_search_results(
    search_results: list[dict[str, Any]], contents: list[dict[str, Any]]
) -> str:
    """Format search results and file contents as structured text."""
    lines = ["## Search Results\n"]
    for i, result in enumerate(search_results):
        path = result.get("path", "unknown")
        lines.append(f"### {path}")
        if i < len(contents):
            content = contents[i].get("content", "")
            lines.append(f"```\n{content}\n```")
        lines.append("")
    return "\n".join(lines)


def _format_session_review(
    session: dict[str, Any],
    messages: list[dict[str, Any]],
    diff: dict[str, Any],
    todos: list[dict[str, Any]] | None = None,
) -> str:
    """Format session review data as structured text."""
    lines = [f"## Session: {session.get('id', 'unknown')}"]
    lines.append(f"Status: {session.get('status', 'unknown')}\n")
    lines.append("### Messages")
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"**{role}:** {content}")
    lines.append("")
    diff_text = diff.get("diff", "")
    if diff_text:
        lines.append("### Diff")
        lines.append(f"```diff\n{diff_text}\n```")
    if todos:
        lines.append("### Todos")
        for todo in todos:
            status = "✓" if todo.get("completed") else "○"
            lines.append(f"- {status} {todo.get('text', '')}")
    return "\n".join(lines)


async def opencode_investigate(
    *,
    query: str,
    max_files: int = 5,
    ctx: Context | None = None,
) -> str:
    """Search project files and optionally analyze results.

    Chains GET /find → GET /file/content for up to max_files results.
    If ctx.sample() is available, returns AI-analyzed results.
    Otherwise returns raw search results + file contents.
    """
    max_files = min(max(max_files, 1), 50)  # clamp to [1, 50]
    from nexus_mcp.http_client import get_http_client

    client = get_http_client()
    search_results = await client.get("/find", params={"query": query})
    if not isinstance(search_results, list):
        search_results = []
    contents: list[dict[str, Any]] = []
    for result in search_results[:max_files]:
        path = result.get("path", "")
        if path:
            content = await client.get("/file/content", params={"path": path})
            contents.append(content if isinstance(content, dict) else {"content": str(content)})
    raw = _format_search_results(search_results[:max_files], contents)
    if ctx:
        try:
            sampled = await ctx.sample(
                f"Analyze these search results for: {query}\n\n{raw}",
                system_prompt="Summarize the relevant findings concisely.",
            )
            if sampled.text is not None:
                return sampled.text
        except Exception:
            logger.debug("ctx.sample() failed, returning raw data", exc_info=True)
    return raw


async def opencode_session_review(
    *,
    session_id: str,
    ctx: Context | None = None,
) -> str:
    """Review a session's messages and file changes.

    Chains GET /session/{id} → GET /session/{id}/message → GET /session/{id}/diff
    → GET /session/{id}/todo.
    If ctx.sample() is available, returns AI-summarized review.
    Otherwise returns raw session data.
    """
    if not re.fullmatch(r"ses[a-zA-Z0-9_-]+", session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    from nexus_mcp.http_client import get_http_client

    client = get_http_client()
    session = await client.get(f"/session/{session_id}")
    messages = await client.get(f"/session/{session_id}/message")
    diff = await client.get(f"/session/{session_id}/diff")
    todo = await client.get(f"/session/{session_id}/todo")
    session_dict = session if isinstance(session, dict) else {}
    messages_list = messages if isinstance(messages, list) else []
    diff_dict = diff if isinstance(diff, dict) else {}
    todo_list = todo if isinstance(todo, list) else []
    raw = _format_session_review(session_dict, messages_list, diff_dict, todo_list)
    if ctx:
        try:
            sampled = await ctx.sample(
                f"Summarize this coding session:\n\n{raw}",
                system_prompt="Provide a concise summary of what was done and what changed.",
            )
            if sampled.text is not None:
                return sampled.text
        except Exception:
            logger.debug("ctx.sample() failed, returning raw data", exc_info=True)
    return raw


def register_compound_tools(mcp: FastMCP) -> None:
    """Register compound tools on the FastMCP server."""
    mcp.tool(tags={"workspace"})(opencode_investigate)
    mcp.tool(tags={"workspace"})(opencode_session_review)
