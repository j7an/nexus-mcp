"""Compatibility imports for the FastMCP-backed preference store."""

from nexus_mcp.mcp.preference_store import (
    PREFERENCES_COLLECTION,
    PREFERENCES_KEY,
    TIERS_COLLECTION,
    TIERS_KEY,
    delete_preferences,
    load_model_tiers,
    load_preferences,
    save_model_tiers,
    save_preferences,
)

__all__ = [
    "PREFERENCES_COLLECTION",
    "PREFERENCES_KEY",
    "TIERS_COLLECTION",
    "TIERS_KEY",
    "delete_preferences",
    "load_model_tiers",
    "load_preferences",
    "save_model_tiers",
    "save_preferences",
]
