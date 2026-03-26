"""Heuristic model tier classification.

Classifies AI models into capability tiers (quick, standard, thorough)
based on vendor naming conventions. This is a lightweight, zero-dependency
classifier -- no network, no sampling, no context required.

The client LLM makes the final runner+model decision using richer inputs
(training knowledge, sampling, live benchmarks). Server-side tiers are
a hint, not authority.
"""

import re
from typing import Literal

Tier = Literal["quick", "standard", "thorough"]

_QUICK_SIGNALS = frozenset({"flash", "lite", "mini", "small", "nano", "instant", "haiku"})
_THOROUGH_SIGNALS = frozenset({"pro", "opus", "max", "large", "ultra"})

_tier_cache: dict[str, Tier] = {}


def _tokenize(model: str) -> frozenset[str]:
    """Split model name into lowercase tokens on non-alphanumeric boundaries."""
    return frozenset(re.split(r"[^a-z0-9]+", model.lower()))


def get_model_tier(model: str) -> Tier:
    """Classify a model's capability tier from name heuristics.

    Resolution: check cache -> scan for signals -> default to 'standard'.
    Results are cached in-memory for the process lifetime.
    """
    if model in _tier_cache:
        return _tier_cache[model]

    tokens = _tokenize(model)
    tier: Tier = "standard"

    if tokens & _QUICK_SIGNALS:
        tier = "quick"
    elif tokens & _THOROUGH_SIGNALS:
        tier = "thorough"

    _tier_cache[model] = tier
    return tier


def get_models_for_tier(models: list[str], tier: Tier) -> list[str]:
    """Filter models matching a given tier."""
    return [m for m in models if get_model_tier(m) == tier]
