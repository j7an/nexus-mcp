# tests/unit/test_sse.py
"""Tests for the SSE stream parser."""

from collections.abc import AsyncIterator

from nexus_mcp.sse import SSEEvent, parse_sse_stream


async def _lines(text: str) -> AsyncIterator[str]:
    """Convert a multi-line string into an async line iterator."""
    for line in text.splitlines(keepends=True):
        yield line


async def _collect(text: str) -> list[SSEEvent]:
    """Parse SSE text and collect all events into a list."""
    return [event async for event in parse_sse_stream(_lines(text))]


class TestParseSSEStream:
    """Tests for parse_sse_stream."""

    async def test_single_event(self):
        raw = "data: hello world\n\n"
        events = await _collect(raw)
        assert len(events) == 1
        assert events[0] == SSEEvent(event="message", data="hello world", id=None)

    async def test_multi_line_data(self):
        raw = "data: line1\ndata: line2\n\n"
        events = await _collect(raw)
        assert len(events) == 1
        assert events[0].data == "line1\nline2"

    async def test_custom_event_type(self):
        raw = "event: status\ndata: ok\n\n"
        events = await _collect(raw)
        assert events[0].event == "status"
        assert events[0].data == "ok"

    async def test_event_with_id(self):
        raw = "id: 42\ndata: payload\n\n"
        events = await _collect(raw)
        assert events[0].id == "42"
        assert events[0].data == "payload"

    async def test_multiple_events(self):
        raw = "data: first\n\ndata: second\n\n"
        events = await _collect(raw)
        assert len(events) == 2
        assert events[0].data == "first"
        assert events[1].data == "second"

    async def test_comment_lines_ignored(self):
        raw = ": this is a comment\ndata: actual\n\n"
        events = await _collect(raw)
        assert len(events) == 1
        assert events[0].data == "actual"

    async def test_empty_data_field(self):
        raw = "data:\n\n"
        events = await _collect(raw)
        assert events[0].data == ""

    async def test_data_with_colon_in_value(self):
        raw = "data: key: value\n\n"
        events = await _collect(raw)
        assert events[0].data == "key: value"

    async def test_no_events_from_empty_input(self):
        raw = ""
        events = await _collect(raw)
        assert events == []

    async def test_incomplete_event_no_blank_line(self):
        """Data without a trailing blank line is not emitted."""
        raw = "data: incomplete"
        events = await _collect(raw)
        assert events == []

    async def test_all_fields_combined(self):
        raw = "id: 7\nevent: update\ndata: payload\n\n"
        events = await _collect(raw)
        assert events[0] == SSEEvent(event="update", data="payload", id="7")

    async def test_unknown_fields_ignored(self):
        raw = "retry: 3000\ndata: ok\n\n"
        events = await _collect(raw)
        assert len(events) == 1
        assert events[0].data == "ok"

    async def test_multiple_blank_lines_between_events(self):
        raw = "data: first\n\n\n\ndata: second\n\n"
        events = await _collect(raw)
        assert len(events) == 2
