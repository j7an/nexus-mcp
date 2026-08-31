"""Platform-specific paths for durable Nexus job state."""

import os
import sys
from pathlib import Path

__all__ = ["default_database_path"]


def default_database_path() -> Path:
    """Return the configured or platform-standard per-user SQLite path."""
    if override := os.environ.get("NEXUS_DB_PATH"):
        return Path(override).expanduser()
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return root / "nexus-mcp" / "nexus.sqlite3"
