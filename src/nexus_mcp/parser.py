# src/nexus_mcp/parser.py
"""Generic JSON extraction utilities for CLI runner output parsing.

Functions here operate on raw strings and have no coupling to any specific CLI agent.
They handle the common pattern of extracting valid JSON from noisy stdout/stderr that
may contain log lines, Node.js warnings, stack traces, or other non-JSON content.
"""

import contextlib
import json
from typing import Any


def _find_balanced_span(
    text: str, open_char: str, close_char: str, search_end: int
) -> tuple[int, int] | None:
    """Find the rightmost balanced bracket span ending at or before search_end.

    Scans right-to-left from the last close_char position, tracking bracket
    depth to locate the matching open_char. Returns (start, end) indices of
    the span (both inclusive), or None if no balanced span is found.
    """
    last_close = text.rfind(close_char, 0, search_end)
    if last_close == -1:
        return None

    depth = 0
    for i in range(last_close, -1, -1):
        if text[i] == close_char:
            depth += 1
        elif text[i] == open_char:
            depth -= 1
            if depth == 0:
                return (i, last_close)

    return None


def extract_last_json_object(text: str) -> dict[str, Any] | None:
    """Find and parse the last JSON object in a multi-line string.

    Uses brace-depth matching (no regex) to locate the last complete {...}
    block. Handles CLI patterns of appending a JSON error block at the end of
    stderr that may contain log lines and stack traces.

    Scans rightward-to-left through '}' positions. If the first candidate
    fails to parse (e.g. '}' appears inside a JSON string value), retries
    with the next candidate — the same retry strategy used by extract_last_json_array.

    Args:
        text: String that may contain JSON, possibly mixed with other content.

    Returns:
        Parsed dict if a valid JSON object is found, None otherwise.
    """
    if not text:
        return None

    search_end = len(text)
    while True:
        span = _find_balanced_span(text, "{", "}", search_end)
        if span is None:
            return None

        start, last_close = span
        with contextlib.suppress(json.JSONDecodeError, ValueError, RecursionError):
            parsed = json.loads(text[start : last_close + 1])
            if isinstance(parsed, dict):
                return parsed

        search_end = last_close


def parse_ndjson_events(stdout: str) -> str | None:
    """Extract agent message text from Codex CLI NDJSON output.

    Codex emits one JSON event object per line. This function collects text
    from ``item.completed`` events where ``item.type == "agent_message"``,
    joining them with ``"\\n\\n"``.

    Text extraction precedence (per event):
        1. ``item["text"]``  — direct text field
        2. ``item["content"][*]["text"]``  — content block list

    Skips blank lines, non-JSON lines, and non-agent-message events silently
    (parser contract: return None on failure, never raise).

    Args:
        stdout: Raw NDJSON output from Codex CLI.

    Returns:
        Joined text from all agent_message events, or None if nothing found.

    Example NDJSON input::

        {"type": "thread.started", "thread_id": "t1"}
        {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "hi"}}
        {"type": "turn.completed"}
    """
    parts: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
            continue
        # Fall back to content block list
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_text = block.get("text")
                    if isinstance(block_text, str):
                        parts.append(block_text)
    return "\n\n".join(parts) if parts else None


def extract_last_json_list(text: str) -> list[Any] | None:
    """Find and parse the last JSON array in text, returning the full list.

    Unlike extract_last_json_array() which returns only the first dict element,
    this returns the entire parsed list — needed by ClaudeRunner to iterate
    all conversation elements and find the result.

    Returns None for empty arrays (same contract as extract_last_json_array).

    Args:
        text: String that may contain a JSON array, possibly mixed with other content.

    Returns:
        Parsed list if a non-empty JSON array is found, None otherwise.
    """
    if not text:
        return None

    search_end = len(text)
    while True:
        span = _find_balanced_span(text, "[", "]", search_end)
        if span is None:
            return None
        start, last_close = span
        with contextlib.suppress(json.JSONDecodeError, ValueError, RecursionError):
            parsed = json.loads(text[start : last_close + 1])
            if isinstance(parsed, list) and parsed:
                return parsed
        search_end = last_close


def extract_last_json_array(text: str) -> dict[str, Any] | None:
    """Find and parse the last JSON array in a multi-line string, returning its first element.

    Scans rightward-to-left through ']' positions using bracket-depth matching.
    Handles CLI GaxiosError format where errors are wrapped in an array:
    [{"error": {...}}]. Skips over ']' characters that appear inside string values
    (e.g. "[object Object]" in session summaries) by trying each candidate in turn.

    Args:
        text: String that may contain a JSON array, possibly mixed with other content.

    Returns:
        First element of the parsed array if it's a dict, None otherwise.
    """
    if not text:
        return None

    search_end = len(text)
    while True:
        span = _find_balanced_span(text, "[", "]", search_end)
        if span is None:
            return None

        start, last_close = span
        with contextlib.suppress(json.JSONDecodeError, ValueError, RecursionError):
            parsed = json.loads(text[start : last_close + 1])
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]

        search_end = last_close
