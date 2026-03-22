"""Embedded SVG icons for the nexus-mcp server and its tools.

Icons are base64-encoded data URIs so they work offline, behind firewalls,
and when installed via PyPI (no external URL dependencies).
"""

# Icons sourced from Lucide (https://lucide.dev)
# Copyright (c) for Lucide Contributors
# Licensed under the ISC License: https://github.com/lucide-icons/lucide/blob/main/LICENSE

from mcp.types import Icon

_MIME = "image/svg+xml"
_PREFIX = "data:image/svg+xml;base64,"

# Lucide "workflow" — two connected boxes representing a router/nexus
_WORKFLOW_B64 = (
    "PHN2ZwogIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICB3aWR0aD0iMjQiCiAgaGVpZ2h0"
    "PSIyNCIKICB2aWV3Qm94PSIwIDAgMjQgMjQiCiAgZmlsbD0ibm9uZSIKICBzdHJva2U9ImN1cnJlbnRDb2xv"
    "ciIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9p"
    "bj0icm91bmQiCj4KICA8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiB4PSIzIiB5PSIzIiByeD0iMiIgLz4K"
    "ICA8cGF0aCBkPSJNNyAxMXY0YTIgMiAwIDAgMCAyIDJoNCIgLz4KICA8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0"
    "PSI4IiB4PSIxMyIgeT0iMTMiIHJ4PSIyIiAvPgo8L3N2Zz4="
)

# Lucide "terminal" — command prompt representing CLI execution
_TERMINAL_B64 = (
    "PHN2ZwogIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICB3aWR0aD0iMjQiCiAgaGVpZ2h0"
    "PSIyNCIKICB2aWV3Qm94PSIwIDAgMjQgMjQiCiAgZmlsbD0ibm9uZSIKICBzdHJva2U9ImN1cnJlbnRDb2xv"
    "ciIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9p"
    "bj0icm91bmQiCj4KICA8cGF0aCBkPSJNMTIgMTloOCIgLz4KICA8cGF0aCBkPSJtNCAxNyA2LTYtNi02IiAv"
    "Pgo8L3N2Zz4="
)

# Lucide "settings" — gear icon representing session preferences
_SETTINGS_B64 = (
    "PHN2ZwogIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICB3aWR0aD0iMjQiCiAgaGVpZ2h0"
    "PSIyNCIKICB2aWV3Qm94PSIwIDAgMjQgMjQiCiAgZmlsbD0ibm9uZSIKICBzdHJva2U9ImN1cnJlbnRDb2xv"
    "ciIKICBzdHJva2Utd2lkdGg9IjIiCiAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogIHN0cm9rZS1saW5lam9p"
    "bj0icm91bmQiCj4KICA8cGF0aCBkPSJNOS42NzEgNC4xMzZhMi4zNCAyLjM0IDAgMCAxIDQuNjU5IDAgMi4z"
    "NCAyLjM0IDAgMCAwIDMuMzE5IDEuOTE1IDIuMzQgMi4zNCAwIDAgMSAyLjMzIDQuMDMzIDIuMzQgMi4zNCAw"
    "IDAgMCAwIDMuODMxIDIuMzQgMi4zNCAwIDAgMS0yLjMzIDQuMDMzIDIuMzQgMi4zNCAwIDAgMC0zLjMxOSAx"
    "LjkxNSAyLjM0IDIuMzQgMCAwIDEtNC42NTkgMCAyLjM0IDIuMzQgMCAwIDAtMy4zMi0xLjkxNSAyLjM0IDIu"
    "MzQgMCAwIDEtMi4zMy00LjAzMyAyLjM0IDIuMzQgMCAwIDAgMC0zLjgzMUEyLjM0IDIuMzQgMCAwIDEgNi4z"
    "NSA2LjA1MWEyLjM0IDIuMzQgMCAwIDAgMy4zMTktMS45MTUiIC8+CiAgPGNpcmNsZSBjeD0iMTIiIGN5PSIx"
    "MiIgcj0iMyIgLz4KPC9zdmc+Cg=="
)

SERVER_ICONS: list[Icon] = [
    Icon(src=f"{_PREFIX}{_WORKFLOW_B64}", mimeType=_MIME),
]

TOOL_EXEC_ICONS: list[Icon] = [
    Icon(src=f"{_PREFIX}{_TERMINAL_B64}", mimeType=_MIME),
]

TOOL_CONFIG_ICONS: list[Icon] = [
    Icon(src=f"{_PREFIX}{_SETTINGS_B64}", mimeType=_MIME),
]
