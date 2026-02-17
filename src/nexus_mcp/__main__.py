# src/nexus_mcp/__main__.py
"""Entry point for running nexus-mcp as a module.

Usage:
    uv run python -m nexus_mcp
"""

from nexus_mcp.server import mcp

if __name__ == "__main__":
    mcp.run()
