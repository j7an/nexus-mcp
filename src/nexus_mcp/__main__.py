# src/nexus_mcp/__main__.py
"""Entry point for running nexus-mcp as a module.

Usage:
    uv run python -m nexus_mcp
"""

import logging
import sys

from nexus_mcp.server import mcp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)

if __name__ == "__main__":
    mcp.run(transport="stdio")
