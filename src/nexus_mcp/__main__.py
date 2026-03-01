# src/nexus_mcp/__main__.py
"""Entry point for running nexus-mcp as a module or console script.

Usage:
    uv run python -m nexus_mcp
    uvx nexus-mcp
"""

import logging
import sys


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    from nexus_mcp.server import mcp

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
