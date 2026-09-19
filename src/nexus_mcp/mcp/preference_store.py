"""In-process preference and model-tier store for nexus-mcp.

Values are global to the server process and persist across MCP sessions for
the process lifetime. They are not written to disk.
"""

import copy
from typing import Any, cast

from fastmcp import Context

PREFERENCES_COLLECTION = "nexus_preferences"
PREFERENCES_KEY = "preferences"

TIERS_COLLECTION = "nexus_tiers"
TIERS_KEY = "model_tiers"

# ponytail: in-process only; move to the Nexus SQLite store keyed by
# principal/workspace when prompt/batch_prompt route through JobService.
_STORE: dict[tuple[str, str], dict[str, Any]] = {}


def reset_store() -> None:
    """Clear all stored values (for tests)."""
    _STORE.clear()


async def _load(ctx: Context, *, key: str, collection: str) -> dict[str, Any] | None:
    """Load a copy of a stored value, returning None if absent."""
    del ctx
    value = _STORE.get((collection, key))
    return None if value is None else copy.deepcopy(value)


async def _save(ctx: Context, value: dict[str, Any], *, key: str, collection: str) -> None:
    """Store a copy of a value, overwriting any existing entry."""
    del ctx
    _STORE[(collection, key)] = copy.deepcopy(value)


async def _delete(ctx: Context, *, key: str, collection: str) -> None:
    """Delete a stored value if present."""
    del ctx
    _STORE.pop((collection, key), None)


async def load_preferences(ctx: Context) -> dict[str, Any] | None:
    """Load preferences from persistent store.

    Returns None if no preferences have been saved.
    """
    return await _load(ctx, key=PREFERENCES_KEY, collection=PREFERENCES_COLLECTION)


async def save_preferences(ctx: Context, prefs_dict: dict[str, Any]) -> None:
    """Save preferences to persistent store."""
    await _save(ctx, prefs_dict, key=PREFERENCES_KEY, collection=PREFERENCES_COLLECTION)


async def delete_preferences(ctx: Context) -> None:
    """Delete preferences from persistent store."""
    await _delete(ctx, key=PREFERENCES_KEY, collection=PREFERENCES_COLLECTION)


async def load_model_tiers(ctx: Context) -> dict[str, str] | None:
    """Load saved model tier classifications from the backing store.

    Returns None if no tiers have been saved yet.
    """
    result = await _load(ctx, key=TIERS_KEY, collection=TIERS_COLLECTION)
    return cast("dict[str, str] | None", result)


async def save_model_tiers(ctx: Context, tiers: dict[str, str]) -> None:
    """Save model tier classifications to the backing store.

    Overwrites any previously saved tiers entirely.
    """
    await _save(ctx, tiers, key=TIERS_KEY, collection=TIERS_COLLECTION)
