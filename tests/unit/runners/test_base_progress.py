# tests/unit/runners/test_base_progress.py
"""Tests for ProgressEmitter protocol and runner progress reporting."""

from unittest.mock import AsyncMock

from nexus_mcp.types import ProgressEmitter  # noqa: TC001


class TestProgressEmitterProtocol:
    """Verify ProgressEmitter protocol is satisfied by async callables."""

    def test_async_callable_satisfies_protocol(self):
        """An async callable with (float, float, str) signature satisfies ProgressEmitter."""
        mock: ProgressEmitter = AsyncMock()
        assert callable(mock)

    def test_noop_progress_satisfies_protocol(self):
        """The _noop_progress default satisfies ProgressEmitter."""
        from nexus_mcp.runners.base import _noop_progress

        emitter: ProgressEmitter = _noop_progress
        assert callable(emitter)
