"""Platform path and private database-file behavior."""

import os
import stat
from pathlib import Path

from nexus_mcp.jobs import paths
from nexus_mcp.jobs.paths import default_database_path
from nexus_mcp.jobs.sqlite_store import SQLiteJobStore


def test_environment_override_takes_precedence(monkeypatch, tmp_path):
    """An explicit database path must override every platform default."""
    override = tmp_path / "custom" / "jobs.db"
    monkeypatch.setenv("NEXUS_DB_PATH", str(override))
    monkeypatch.setattr(paths.sys, "platform", "darwin")

    assert default_database_path() == override


def test_macos_default_path(monkeypatch, tmp_path):
    """macOS data lives below the per-user Application Support directory."""
    monkeypatch.delenv("NEXUS_DB_PATH", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)

    assert default_database_path() == (
        tmp_path / "Library" / "Application Support" / "nexus-mcp" / "nexus.sqlite3"
    )


def test_windows_default_path_uses_local_app_data(monkeypatch, tmp_path):
    """Windows data honors the user's LOCALAPPDATA directory."""
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.delenv("NEXUS_DB_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)

    assert default_database_path() == local_app_data / "nexus-mcp" / "nexus.sqlite3"


def test_windows_empty_local_app_data_uses_home_fallback(monkeypatch, tmp_path):
    """An empty LOCALAPPDATA value never turns the durable database into a cwd path."""
    monkeypatch.delenv("NEXUS_DB_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", "")
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)

    assert default_database_path() == (
        tmp_path / "AppData" / "Local" / "nexus-mcp" / "nexus.sqlite3"
    )


def test_linux_default_path_uses_xdg_data_home(monkeypatch, tmp_path):
    """Unix data honors XDG_DATA_HOME when configured."""
    xdg_data_home = tmp_path / "xdg"
    monkeypatch.delenv("NEXUS_DB_PATH", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data_home))
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)

    assert default_database_path() == xdg_data_home / "nexus-mcp" / "nexus.sqlite3"


def test_linux_default_path_falls_back_below_home(monkeypatch, tmp_path):
    """Unix data has a stable per-user fallback without XDG_DATA_HOME."""
    monkeypatch.delenv("NEXUS_DB_PATH", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)

    assert default_database_path() == tmp_path / ".local/share/nexus-mcp/nexus.sqlite3"


def test_linux_empty_xdg_data_home_falls_back_below_home(monkeypatch, tmp_path):
    """An empty XDG_DATA_HOME value never turns the durable database into a cwd path."""
    monkeypatch.delenv("NEXUS_DB_PATH", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", "")
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setattr(paths.Path, "home", lambda: tmp_path)

    assert default_database_path() == tmp_path / ".local/share/nexus-mcp/nexus.sqlite3"


async def test_open_creates_private_database_files(tmp_path):
    """Opening the store removes group and other access from every SQLite file."""
    database_path = tmp_path / "private" / "nexus.sqlite3"
    store = SQLiteJobStore(database_path)

    await store.open()
    try:
        assert stat.S_IMODE(database_path.parent.stat().st_mode) & 0o077 == 0
        if os.name == "posix":
            sqlite_paths = (
                store.path,
                Path(f"{store.path}-wal"),
                Path(f"{store.path}-shm"),
            )
            assert all(path.exists() for path in sqlite_paths)
            assert all(stat.S_IMODE(path.stat().st_mode) & 0o077 == 0 for path in sqlite_paths)
    finally:
        await store.close()
