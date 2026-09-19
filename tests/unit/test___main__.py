"""Tests for the Python version check in __main__.py."""

import re
from importlib.metadata import metadata
from unittest.mock import patch

import pytest

from nexus_mcp.__main__ import _check_python_version


def _minimum_python() -> str:
    """Return the ``major.minor`` floor declared in installed package metadata."""
    match = re.search(r">=\s*(\d+\.\d+)", metadata("nexus-mcp")["Requires-Python"])
    assert match, "Requires-Python must declare a >= floor"
    return match.group(1)


class TestCheckPythonVersion:
    """Tests for _check_python_version()."""

    def test_exits_on_old_python(self):
        """Simulates Python 3.11 (below the declared floor) to trigger the version warning."""
        fake_version = (3, 11, 0, "final", 0)
        with (
            patch.object(__import__("sys"), "version_info", fake_version),
            pytest.raises(SystemExit) as exc_info,
        ):
            _check_python_version()

        message = str(exc_info.value)
        assert f"requires Python {_minimum_python()}+" in message
        assert "running Python 3.11" in message
        assert "uvx nexus-mcp" in message

    def test_passes_on_current_python(self):
        """No exit when running on a supported Python version."""
        _check_python_version()  # Should not raise

    def test_passes_when_no_requires_python(self):
        """Gracefully handles missing Requires-Python metadata."""
        with patch("importlib.metadata.metadata", return_value={}):
            _check_python_version()  # Should not raise
