"""Tests for request correlation ID propagation."""

import logging

from nexus_mcp.correlation import CorrelationFilter, correlation_id, set_correlation_id


class TestCorrelationId:
    """Tests for correlation_id ContextVar."""

    def test_default_is_dash(self):
        """No correlation ID set returns '-'."""
        assert correlation_id.get() == "-"

    def test_set_and_get(self):
        """set_correlation_id stores value retrievable via .get()."""
        token = set_correlation_id()
        value = correlation_id.get()
        assert len(value) == 8
        assert value != "-"
        correlation_id.reset(token)

    def test_reset_restores_default(self):
        """Resetting token restores default '-'."""
        token = set_correlation_id()
        correlation_id.reset(token)
        assert correlation_id.get() == "-"


class TestCorrelationFilter:
    """Tests for CorrelationFilter logging filter."""

    def test_adds_req_id_to_record(self):
        """Filter injects req_id attribute into log records."""
        token = set_correlation_id()
        filt = CorrelationFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        result = filt.filter(record)
        assert result is True
        assert hasattr(record, "req_id")
        assert record.req_id == correlation_id.get()  # type: ignore[attr-defined]
        correlation_id.reset(token)

    def test_default_req_id_is_dash(self):
        """When no correlation ID is set, req_id is '-'."""
        filt = CorrelationFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        filt.filter(record)
        assert record.req_id == "-"  # type: ignore[attr-defined]


class TestEmitterCorrelation:
    """Verify emitter logs carry correlation context."""

    async def test_mcp_emitter_logs_carry_req_id(self, caplog):
        """LogEmitter's Python logger calls inherit the correlation ContextVar."""
        from unittest.mock import AsyncMock

        from nexus_mcp.correlation import CorrelationFilter
        from nexus_mcp.emitters import make_mcp_emitter

        mock_ctx = AsyncMock()
        mock_ctx.info = AsyncMock()
        mock_ctx.debug = AsyncMock()
        mock_ctx.warning = AsyncMock()
        mock_ctx.error = AsyncMock()
        emitter = make_mcp_emitter(mock_ctx)

        emitter_logger = logging.getLogger("nexus_mcp.emitters")
        filt = CorrelationFilter()
        emitter_logger.addFilter(filt)

        token = set_correlation_id()
        expected_id = correlation_id.get()
        try:
            with caplog.at_level(logging.INFO, logger="nexus_mcp.emitters"):
                await emitter("info", "test message")
            for record in caplog.records:
                if record.name == "nexus_mcp.emitters":
                    assert record.req_id == expected_id  # type: ignore[attr-defined]
        finally:
            correlation_id.reset(token)
            emitter_logger.removeFilter(filt)
