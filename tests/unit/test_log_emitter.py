# tests/unit/test_log_emitter.py
"""Tests for the default LogEmitter implementation."""

from unittest.mock import patch

import pytest

from nexus_mcp.runners.base import _default_log_emitter


class TestDefaultLogEmitter:
    """_default_log_emitter routes each level to the correct logger method."""

    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
    async def test_routes_to_correct_logger_method(self, level):
        with patch("nexus_mcp.runners.base.logger") as mock_logger:
            await _default_log_emitter(level, "test message")
            getattr(mock_logger, level).assert_called_once_with("test message")

    async def test_does_not_call_other_levels(self):
        with patch("nexus_mcp.runners.base.logger") as mock_logger:
            await _default_log_emitter("warning", "test message")
            mock_logger.info.assert_not_called()
            mock_logger.error.assert_not_called()
            mock_logger.debug.assert_not_called()
