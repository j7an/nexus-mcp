"""Unit tests for the in-process preference and tier store."""

from unittest.mock import AsyncMock

import pytest
from fastmcp import Context

from nexus_mcp.store import (
    delete_preferences,
    load_model_tiers,
    load_preferences,
    reset_store,
    save_model_tiers,
    save_preferences,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def ctx() -> AsyncMock:
    return AsyncMock(spec=Context)


async def test_load_returns_none_when_empty(ctx):
    assert await load_preferences(ctx) is None
    assert await load_model_tiers(ctx) is None


async def test_save_then_load_round_trips(ctx):
    await save_preferences(ctx, {"execution_mode": "yolo", "model": None})
    assert await load_preferences(ctx) == {"execution_mode": "yolo", "model": None}


async def test_values_persist_across_contexts(ctx):
    await save_preferences(ctx, {"model": "m"})
    assert await load_preferences(AsyncMock(spec=Context)) == {"model": "m"}


async def test_stored_value_is_isolated_from_caller_mutation(ctx):
    prefs = {"model": "m"}
    await save_preferences(ctx, prefs)
    prefs["model"] = "changed"
    loaded = await load_preferences(ctx)
    assert loaded == {"model": "m"}
    loaded["model"] = "changed-again"
    assert await load_preferences(ctx) == {"model": "m"}


async def test_delete_preferences_leaves_tiers(ctx):
    await save_preferences(ctx, {"model": "m"})
    await save_model_tiers(ctx, {"m": "fast"})
    await delete_preferences(ctx)
    assert await load_preferences(ctx) is None
    assert await load_model_tiers(ctx) == {"m": "fast"}


async def test_delete_missing_is_noop(ctx):
    await delete_preferences(ctx)
