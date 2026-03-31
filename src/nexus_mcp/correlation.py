"""Request correlation ID for concurrent tool call log disambiguation.

Uses a contextvars.ContextVar so the ID propagates through async/await
chains without modifying function signatures. The CorrelationFilter
injects ``req_id`` into every log record for the formatter to consume.

Usage:
    token = set_correlation_id()     # outermost middleware sets this
    ...                              # all downstream logs include req_id
    correlation_id.reset(token)      # cleanup after tool call completes
"""

__all__ = ["CorrelationFilter", "correlation_id", "set_correlation_id"]

import logging
import uuid
from contextvars import ContextVar, Token

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id() -> Token[str]:
    """Generate a short request ID and store it in the ContextVar.

    Returns the token for resetting after the request completes.
    Uses first 8 chars of uuid4 — sufficient for log disambiguation
    (collision probability negligible within a single server session).
    """
    return correlation_id.set(uuid.uuid4().hex[:8])


class CorrelationFilter(logging.Filter):
    """Logging filter that injects ``req_id`` into every log record.

    Attach to a handler or root logger. The formatter can then use
    ``%(req_id)s`` to include the correlation ID in output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.req_id = correlation_id.get()
        return True
