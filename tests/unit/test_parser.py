# tests/unit/test_parser.py
"""Tests for nexus_mcp.parser module.

Tests verify:
- extract_last_json_object(): brace-depth JSON extraction from mixed text
- extract_last_json_array(): bracket-depth array extraction, returns first dict element
- extract_last_json_list(): bracket-depth array extraction, returns full list
"""

from nexus_mcp.parser import (
    extract_last_json_array,
    extract_last_json_list,
    extract_last_json_object,
    parse_ndjson_events,
)


class TestExtractLastJsonObject:
    """Test extract_last_json_object() finds the last JSON object in mixed text."""

    def test_extracts_json_from_end_of_mixed_text(self):
        """Extracts the JSON object appended at the end of a mixed-content string."""
        text = 'log line 1\nlog line 2\n{"key": "value"}'
        result = extract_last_json_object(text)
        assert result == {"key": "value"}

    def test_returns_none_for_no_json(self):
        """Returns None when the text contains no JSON object."""
        result = extract_last_json_object("just plain text with no JSON")
        assert result is None

    def test_returns_none_for_empty_string(self):
        """Returns None for empty string input."""
        result = extract_last_json_object("")
        assert result is None

    def test_handles_nested_braces(self):
        """Correctly handles JSON with nested objects (brace-depth matching)."""
        text = 'preamble\n{"outer": {"inner": "value", "count": 42}}'
        result = extract_last_json_object(text)
        assert result == {"outer": {"inner": "value", "count": 42}}

    def test_picks_last_json_when_multiple_present(self):
        """Returns the last JSON object when multiple are present in the text."""
        text = '{"first": 1}\nsome text\n{"second": 2}'
        result = extract_last_json_object(text)
        assert result == {"second": 2}

    def test_ignores_non_dict_json(self):
        """Returns None when the extracted JSON is not a dict (e.g., an array)."""
        text = "text before\n[1, 2, 3]"
        result = extract_last_json_object(text)
        assert result is None

    def test_known_limitation_braces_in_string_values(self):
        """Documents known limitation: brace chars in string values confuse depth tracking.

        Balanced brace chars (e.g. '{foo}') cancel out and accidentally find the right
        boundary; unbalanced ones (e.g. lone '}') cause json.loads to fail and return None.
        In practice, Google API error messages rarely contain literal brace characters.
        """
        # Balanced braces in string values — algorithm accidentally gets the right answer
        text = 'log\n{"key": "{balanced}"}'
        result = extract_last_json_object(text)
        assert result == {"key": "{balanced}"}

        # Unbalanced closing brace in string value — algorithm mis-identifies boundary,
        # json.loads fails, returns None (falls through to generic SubprocessError)
        text_unbalanced = 'log\n{"key": "has a } brace"}'
        result_unbalanced = extract_last_json_object(text_unbalanced)
        assert result_unbalanced is None

    def test_balanced_braces_but_invalid_json_returns_none(self):
        """Returns None when brace-matching finds a span but json.loads rejects the content.

        '{not: valid json}' has balanced outer braces so _find_balanced_span succeeds,
        but unquoted keys make json.loads raise JSONDecodeError (lines 63-64).
        """
        result = extract_last_json_object("{not: valid json}")
        assert result is None

    def test_deeply_nested_braces_no_recursion_error(self):
        """200 levels of nesting does not raise RecursionError.

        The iterative _find_balanced_span retries inward span-by-span until it
        reaches the innermost '{}', which json.loads parses as the empty dict {}.
        RecursionError is also suppressed explicitly, but the algorithm never
        reaches that depth for iterative code.
        """
        deeply_nested = "{" * 200 + "}" * 200
        result = extract_last_json_object(deeply_nested)
        assert result == {}


class TestExtractLastJsonArray:
    """Test extract_last_json_array() finds the last JSON array and returns its first dict."""

    def test_extracts_first_element_from_gaxios_error_array(self):
        """Extracts the first dict from a GaxiosError array embedded in mixed text."""
        text = (
            "Gemini CLI error log\n"
            '[{"error": {"code": 429, "message": "No capacity available"}}]\n'
            "additional log line"
        )
        result = extract_last_json_array(text)
        assert result == {"error": {"code": 429, "message": "No capacity available"}}

    def test_returns_none_for_empty_string(self):
        """Returns None for empty string input."""
        result = extract_last_json_array("")
        assert result is None

    def test_returns_none_for_no_array(self):
        """Returns None when text contains no JSON array (bare object only)."""
        result = extract_last_json_array('{"key": "value"}')
        assert result is None

    def test_returns_none_for_non_dict_first_element(self):
        """Returns None when the array's first element is not a dict (e.g., a number)."""
        result = extract_last_json_array("[1, 2, 3]")
        assert result is None

    def test_returns_none_for_empty_array(self):
        """Returns None for an empty JSON array."""
        result = extract_last_json_array("[]")
        assert result is None

    def test_handles_nested_objects_in_array(self):
        """Correctly parses arrays whose elements contain nested objects."""
        text = '[{"error": {"code": 429, "details": [{"type": "quota"}]}}]'
        result = extract_last_json_array(text)
        assert result == {"error": {"code": 429, "details": [{"type": "quota"}]}}

    def test_picks_last_array_when_multiple_present(self):
        """Returns the first element of the last array when multiple arrays exist."""
        text = '[{"first": 1}]\nsome text\n[{"second": 2}]'
        result = extract_last_json_array(text)
        assert result == {"second": 2}

    def test_deeply_nested_brackets_returns_none_without_recursion_error(self):
        """200 levels of nesting returns None cleanly (iterative algorithm, no stack overflow)."""
        deeply_nested = "[" * 200 + "]" * 200
        result = extract_last_json_array(deeply_nested)
        assert result is None


class TestExtractLastJsonList:
    """Test extract_last_json_list() returns the full list from the last JSON array."""

    def test_clean_input_returns_full_list(self):
        """Clean JSON array → full list returned."""
        text = '[{"type": "result", "result": "hello"}, {"type": "assistant"}]'
        result = extract_last_json_list(text)
        assert result == [{"type": "result", "result": "hello"}, {"type": "assistant"}]

    def test_noisy_stdout_extracts_array(self):
        """Log lines before JSON array are ignored; array extracted correctly."""
        text = (
            "(node:1234) Warning: some deprecation\n"
            "Loaded credentials.\n"
            '[{"type": "result", "result": "pong"}]'
        )
        result = extract_last_json_list(text)
        assert result == [{"type": "result", "result": "pong"}]

    def test_empty_string_returns_none(self):
        """Empty string → None."""
        result = extract_last_json_list("")
        assert result is None

    def test_no_brackets_returns_none(self):
        """Text without brackets → None."""
        result = extract_last_json_list('{"key": "value"}')
        assert result is None

    def test_invalid_json_returns_none(self):
        """Brackets found but invalid JSON → None."""
        result = extract_last_json_list("[not valid json}")
        assert result is None

    def test_non_dict_elements_returned_as_list(self):
        """List of scalars returned as-is (no element-type filtering)."""
        result = extract_last_json_list("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_empty_array_returns_none(self):
        """Empty JSON array '[]' → None (same contract as extract_last_json_array)."""
        result = extract_last_json_list("[]")
        assert result is None

    def test_returns_all_elements_not_just_first(self):
        """Full list returned — not just the first dict like extract_last_json_array."""
        text = '[{"a": 1}, {"b": 2}, {"c": 3}]'
        result = extract_last_json_list(text)
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]
        assert len(result) == 3  # type: ignore[arg-type]

    def test_picks_last_array_when_multiple_present(self):
        """Returns the full last array when multiple arrays exist in text."""
        text = '[{"first": 1}]\nsome text\n[{"second": 2}, {"third": 3}]'
        result = extract_last_json_list(text)
        assert result == [{"second": 2}, {"third": 3}]

    def test_deeply_nested_brackets_no_recursion_error(self):
        """200 levels of nesting does not raise RecursionError.

        json.loads('[' * 200 + ']' * 200) returns a 200-deep nested list which Python
        successfully parses. The outermost span yields a truthy (non-empty) list, so
        the function returns a deeply nested list rather than None.
        """
        deeply_nested = "[" * 200 + "]" * 200
        result = extract_last_json_list(deeply_nested)
        assert isinstance(result, list) and len(result) == 1


class TestParseNdjsonEvents:
    """Test parse_ndjson_events() extracts text from Codex NDJSON output."""

    def test_empty_input_returns_none(self):
        """Empty string yields no events → None."""
        assert parse_ndjson_events("") is None

    def test_single_item_text_field(self):
        """Single agent_message event with direct text field."""
        line = '{"type": "item.completed", "item": {"type": "agent_message", "text": "pong"}}'
        assert parse_ndjson_events(line) == "pong"

    def test_multiple_items_joined_with_double_newline(self):
        """Multiple agent_message events are joined with \\n\\n."""
        ndjson = "\n".join(
            [
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "first"}}',
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "second"}}',
            ]
        )
        assert parse_ndjson_events(ndjson) == "first\n\nsecond"

    def test_content_list_fallback(self):
        """Uses content[*].text when direct text field is absent."""
        line = (
            '{"type": "item.completed", "item": {'
            '"type": "agent_message", "content": [{"text": "from content"}]}}'
        )
        assert parse_ndjson_events(line) == "from content"

    def test_non_agent_message_events_skipped(self):
        """Events with item.type != agent_message are silently ignored."""
        ndjson = "\n".join(
            [
                '{"type": "thread.started", "thread_id": "t1"}',
                '{"type": "item.completed", "item": {"type": "tool_call", "text": "skip me"}}',
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "keep"}}',
            ]
        )
        assert parse_ndjson_events(ndjson) == "keep"

    def test_non_json_lines_skipped(self):
        """Non-JSON lines are skipped without error."""
        ndjson = "\n".join(
            [
                "not json at all",
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}',
            ]
        )
        assert parse_ndjson_events(ndjson) == "ok"

    def test_blank_lines_skipped(self):
        """Blank / whitespace-only lines are ignored."""
        ndjson = "\n".join(
            [
                "",
                '{"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}',
                "   ",
            ]
        )
        assert parse_ndjson_events(ndjson) == "hi"

    def test_trailing_newline_handled(self):
        """Trailing newline in stdout doesn't produce an empty part."""
        ndjson = '{"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}\n'
        assert parse_ndjson_events(ndjson) == "hi"

    def test_lifecycle_events_with_no_agent_message_return_none(self):
        """Status/lifecycle-only event stream with no agent_message → None."""
        ndjson = "\n".join(
            [
                '{"type": "thread.started"}',
                '{"type": "turn.completed"}',
            ]
        )
        assert parse_ndjson_events(ndjson) is None

    def test_partial_success_error_and_message(self):
        """Stream with an error event followed by a valid agent_message → message text.

        Error events (type='error') are not item.completed events, so they are
        silently dropped by the filter guard. The agent_message is still returned.
        """
        error_event = '{"type": "error", "message": "something went wrong"}'
        message_event = (
            '{"type": "item.completed", "item": {"type": "agent_message", "text": "ok"}}'
        )
        ndjson = "\n".join([error_event, message_event])
        assert parse_ndjson_events(ndjson) == "ok"
