# tests/unit/test_parser.py
"""Tests for nexus_mcp.parser module.

Tests verify:
- extract_last_json_object(): brace-depth JSON extraction from mixed text
- extract_last_json_array(): bracket-depth array extraction, returns first dict element
"""

from nexus_mcp.parser import extract_last_json_array, extract_last_json_object


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
