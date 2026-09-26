"""Registry for optional execution backends.

A backend module ``nexus_mcp.backends.<name>`` provides asynchronous ``run`` and
``info`` functions. It counts as installed when the SDK for its optional extra
is importable.
"""

import importlib
from importlib.util import find_spec
from types import ModuleType
from typing import cast

from nexus_mcp.types import BackendInfo, BackendName

__all__ = ["NAMES", "get", "info", "install_hint", "installed"]

# Backend name -> top-level package installed by its optional extra.
_SDK_PACKAGES: dict[BackendName, str] = {"claude": "claude_agent_sdk"}
NAMES: tuple[BackendName, ...] = tuple(_SDK_PACKAGES)


def installed(name: BackendName) -> bool:
    """Return whether the backend's SDK extra is importable."""
    return find_spec(_SDK_PACKAGES[name]) is not None


def install_hint(name: BackendName) -> str:
    """Return the command that installs the backend's extra."""
    return (
        f"Install the '{name}' extra: pip install 'nexus-mcp[{name}]' "
        f"(or run uvx --with 'nexus-mcp[{name}]' nexus-mcp)"
    )


def get(name: BackendName) -> ModuleType:
    """Import a backend module. Call only when installed(name) is true."""
    return importlib.import_module(f"nexus_mcp.backends.{name}")


async def info(name: BackendName) -> BackendInfo:
    """Describe one backend, whether or not it is installed."""
    if not installed(name):
        return BackendInfo(name=name, installed=False, hint=install_hint(name))
    return cast("BackendInfo", await get(name).info())
