# src/nexus_mcp/sse.py
"""Server-Sent Events (SSE) stream parser.

Parses an async line iterator into structured SSEEvent objects.
Implements the SSE spec: data/event/id fields, comment lines, blank-line boundaries.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SSEEvent:
    """A single parsed SSE event."""

    event: str = "message"
    data: str = ""
    id: str | None = None


async def parse_sse_stream(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    """Parse an async line stream into SSE events.

    Yields SSEEvent objects as they are completed (delimited by blank lines).
    Handles: data/event/id fields, multi-line data (joined with "\\n"),
    comment lines (: prefix), and unknown fields (ignored).

    Args:
        lines: Async iterator of raw lines (with or without trailing newlines).

    Yields:
        SSEEvent for each complete event (blank-line delimited).
    """
    event_type = "message"
    data_parts: list[str] = []
    event_id: str | None = None
    has_data = False

    async for raw_line in lines:
        line = raw_line.rstrip("\n\r")

        # Blank line = event boundary
        if not line:
            if has_data:
                yield SSEEvent(event=event_type, data="\n".join(data_parts), id=event_id)
            # Reset for next event
            event_type = "message"
            data_parts = []
            event_id = None
            has_data = False
            continue

        # Comment line
        if line.startswith(":"):
            continue

        # Parse field: value
        if ":" in line:
            field, _, value = line.partition(":")
            value = value.removeprefix(" ")  # strip single leading space per SSE spec
        else:
            field = line
            value = ""

        match field:
            case "data":
                data_parts.append(value)
                has_data = True
            case "event":
                event_type = value
            case "id":
                event_id = value
            case _:
                pass  # Unknown fields (e.g., "retry") are ignored
