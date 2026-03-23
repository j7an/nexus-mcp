# tests/unit/test___main__.py
"""Tests for the Python version check in __main__.py."""

from unittest.mock import patch

import pytest

from nexus_mcp.__main__ import _check_python_version


class TestCheckPythonVersion:
    """Tests for _check_python_version()."""

    def test_exits_on_old_python(self):
        """Simulates Python 3.11 to trigger the version warning."""
        fake_version = (3, 11, 0, "final", 0)
        with (
            patch.object(__import__("sys"), "version_info", fake_version),
            pytest.raises(SystemExit) as exc_info,
        ):
            _check_python_version()

        message = str(exc_info.value)
        assert "requires Python 3.13+" in message
        assert "running Python 3.11" in message
        assert "uvx nexus-mcp" in message

    def test_passes_on_current_python(self):
        """No exit when running on a supported Python version."""
        _check_python_version()  # Should not raise

    def test_passes_when_no_requires_python(self):
        """Gracefully handles missing Requires-Python metadata."""
        with patch("importlib.metadata.metadata", return_value={}):
            _check_python_version()  # Should not raise
