"""Persistent store access for nexus-mcp.

Provides access to FastMCP's backing store using custom collections
that bypass session scoping. Data in these collections persists across
MCP sessions for the server's process lifetime (MemoryStore default)
or across restarts (FileTreeStore/RedisStore).
"""

from typing import Any, cast

from fastmcp import Context

PREFERENCES_COLLECTION = "nexus_preferences"
PREFERENCES_KEY = "preferences"

TIERS_COLLECTION = "nexus_tiers"
TIERS_KEY = "model_tiers"


def _get_store(ctx: Context) -> Any:
    """Get the backing store from the MCP server context.

    Returns the PydanticAdapter[StateValue] which wraps the configured
    AsyncKeyValue backend. Using a custom collection bypasses session-scoped
    key prefixing done by ctx.get_state/ctx.set_state.
    """
    return ctx.fastmcp._state_store


async def _load(ctx: Context, *, key: str, collection: str) -> dict[str, Any] | None:
    """Load a value from the backing store, returning None if absent."""
    store = _get_store(ctx)
    data = await store.get(key=key, collection=collection)
    if data is None:
        return None
    return cast("dict[str, Any]", data.value)


async def _save(ctx: Context, value: dict[str, Any], *, key: str, collection: str) -> None:
    """Save a value to the backing store, overwriting any existing entry."""
    store = _get_store(ctx)
    await store.put(key=key, value={"value": value}, collection=collection)


async def _delete(ctx: Context, *, key: str, collection: str) -> None:
    """Delete a value from the backing store."""
    store = _get_store(ctx)
    await store.delete(key=key, collection=collection)


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
