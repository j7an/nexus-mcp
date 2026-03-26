# tests/unit/test_store.py
"""Unit tests for persistent store helpers."""

from unittest.mock import AsyncMock, PropertyMock

from fastmcp import Context

from nexus_mcp.store import (
    PREFERENCES_COLLECTION,
    PREFERENCES_KEY,
    delete_preferences,
    load_preferences,
    save_preferences,
)


def _make_ctx_with_store(store: AsyncMock) -> AsyncMock:
    """Create a mock Context whose fastmcp._state_store returns the given store."""
    ctx = AsyncMock(spec=Context)
    fastmcp_mock = AsyncMock()
    type(fastmcp_mock)._state_store = PropertyMock(return_value=store)
    type(ctx).fastmcp = PropertyMock(return_value=fastmcp_mock)
    return ctx


class TestLoadPreferences:
    async def test_returns_none_when_store_empty(self):
        store = AsyncMock()
        store.get.return_value = None
        ctx = _make_ctx_with_store(store)

        result = await load_preferences(ctx)

        assert result is None
        store.get.assert_awaited_once_with(key=PREFERENCES_KEY, collection=PREFERENCES_COLLECTION)

    async def test_returns_value_from_store(self):
        store = AsyncMock()
        prefs_data = {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
        state_value = AsyncMock()
        state_value.value = prefs_data
        store.get.return_value = state_value
        ctx = _make_ctx_with_store(store)

        result = await load_preferences(ctx)

        assert result == prefs_data


class TestSavePreferences:
    async def test_saves_to_store(self):
        store = AsyncMock()
        ctx = _make_ctx_with_store(store)
        prefs_data = {"execution_mode": "yolo", "model": None}

        await save_preferences(ctx, prefs_data)

        store.put.assert_awaited_once_with(
            key=PREFERENCES_KEY,
            value={"value": prefs_data},
            collection=PREFERENCES_COLLECTION,
        )


class TestDeletePreferences:
    async def test_deletes_from_store(self):
        store = AsyncMock()
        ctx = _make_ctx_with_store(store)

        await delete_preferences(ctx)

        store.delete.assert_awaited_once_with(
            key=PREFERENCES_KEY, collection=PREFERENCES_COLLECTION
        )


class TestRoundTrip:
    async def test_save_then_load_round_trip(self):
        """Save preferences, then load them back — verifies key/collection consistency."""
        store = AsyncMock()
        ctx = _make_ctx_with_store(store)
        prefs_data = {
            "execution_mode": "yolo",
            "model": "gemini-2.5-flash",
            "max_retries": 5,
        }

        await save_preferences(ctx, prefs_data)

        state_value = AsyncMock()
        state_value.value = prefs_data
        store.get.return_value = state_value

        result = await load_preferences(ctx)
        assert result == prefs_data

    async def test_delete_then_load_returns_none(self):
        store = AsyncMock()
        ctx = _make_ctx_with_store(store)

        await delete_preferences(ctx)

        store.get.return_value = None
        result = await load_preferences(ctx)
        assert result is None
