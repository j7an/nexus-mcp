# src/nexus_mcp/mcp/compound_tools.py
"""Compound tools that chain multiple OpenCode HTTP calls.

Each tool aggregates data from multiple API endpoints.
Returns deterministic text built from OpenCode server data.

Tools:
- opencode_investigate: Search + read files
- opencode_session_review: Session + messages + diff
"""

import re
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from nexus_mcp.exceptions import ConfigurationError
from nexus_mcp.http_client import OpenCodeHTTPClient, get_http_client


def _get_tool_http_client() -> OpenCodeHTTPClient:
    """Translate missing OpenCode configuration at the MCP tool boundary."""
    try:
        return get_http_client()
    except ConfigurationError as exc:
        raise ToolError(str(exc)) from exc


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
) -> str:
    """Search project files and return the matching results.

    Chains GET /find → GET /file/content for up to max_files results.
    """
    max_files = min(max(max_files, 1), 50)  # clamp to [1, 50]
    client = _get_tool_http_client()
    search_results = await client.get("/find", params={"query": query})
    if not isinstance(search_results, list):
        search_results = []
    contents: list[dict[str, Any]] = []
    for result in search_results[:max_files]:
        path = result.get("path", "")
        if path:
            content = await client.get("/file/content", params={"path": path})
            contents.append(content if isinstance(content, dict) else {"content": str(content)})
    return _format_search_results(search_results[:max_files], contents)


async def opencode_session_review(
    *,
    session_id: str,
) -> str:
    """Review a session's messages and file changes.

    Chains GET /session/{id} → GET /session/{id}/message → GET /session/{id}/diff
    → GET /session/{id}/todo.
    """
    if not re.fullmatch(r"ses[a-zA-Z0-9_-]+", session_id):
        raise ValueError(f"Invalid session_id: {session_id!r}")
    client = _get_tool_http_client()
    session = await client.get(f"/session/{session_id}")
    messages = await client.get(f"/session/{session_id}/message")
    diff = await client.get(f"/session/{session_id}/diff")
    todo = await client.get(f"/session/{session_id}/todo")
    session_dict = session if isinstance(session, dict) else {}
    messages_list = messages if isinstance(messages, list) else []
    diff_dict = diff if isinstance(diff, dict) else {}
    todo_list = todo if isinstance(todo, list) else []
    return _format_session_review(session_dict, messages_list, diff_dict, todo_list)


def register_compound_tools(mcp: FastMCP) -> None:
    """Register compound tools on the FastMCP server."""
    mcp.tool(tags={"workspace"})(opencode_investigate)
    mcp.tool(tags={"workspace"})(opencode_session_review)
