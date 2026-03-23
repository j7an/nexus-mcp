# src/nexus_mcp/__main__.py
"""Entry point for running nexus-mcp as a module or console script.

Usage:
    uv run python -m nexus_mcp
    uvx nexus-mcp
"""

import sys


def _check_python_version() -> None:
    """Exit early with a clear message if Python version is too old.

    Reads requires-python from installed package metadata so the minimum
    version stays in sync with pyproject.toml automatically.
    """
    import re
    from importlib.metadata import metadata

    requires_python = metadata("nexus-mcp").get("Requires-Python", "")
    match = re.search(r">=\s*(\d+)\.(\d+)", requires_python)
    if not match:
        return
    min_major, min_minor = int(match.group(1)), int(match.group(2))
    if sys.version_info >= (min_major, min_minor):
        return
    sys.exit(
        f"nexus-mcp requires Python {min_major}.{min_minor}+, "
        f"but you are running Python {sys.version_info[0]}.{sys.version_info[1]}.\n"
        "Install with: uvx nexus-mcp  (uv auto-downloads the correct Python)"
    )


_check_python_version()

import logging  # noqa: E402 — must run version check before importing project code


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in ("--version", "-V"):
        from nexus_mcp import __version__

        print(f"nexus-mcp {__version__}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    from nexus_mcp.server import mcp

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
