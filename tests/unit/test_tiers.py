"""Unit tests for model tier heuristic classification."""

from nexus_mcp.tiers import get_model_tier, get_models_for_tier


class TestGetModelTier:
    def test_flash_models_are_quick(self):
        assert get_model_tier("gemini-2.5-flash") == "quick"
        assert get_model_tier("gemini-2.5-flash-lite") == "quick"
        assert get_model_tier("gemini-3-flash-preview") == "quick"

    def test_lite_mini_models_are_quick(self):
        assert get_model_tier("gpt-5.4-mini") == "quick"
        assert get_model_tier("gemini-2.5-flash-lite") == "quick"

    def test_haiku_models_are_quick(self):
        assert get_model_tier("claude-haiku-4-5-20251001") == "quick"

    def test_pro_models_are_thorough(self):
        assert get_model_tier("gemini-2.5-pro") == "thorough"
        assert get_model_tier("gemini-3.1-pro-preview") == "thorough"

    def test_opus_models_are_thorough(self):
        assert get_model_tier("claude-opus-4-6") == "thorough"

    def test_max_models_are_thorough(self):
        assert get_model_tier("gpt-5.1-codex-max") == "thorough"

    def test_unknown_models_default_to_standard(self):
        assert get_model_tier("gpt-5.2") == "standard"
        assert get_model_tier("gpt-5.2-codex") == "standard"
        assert get_model_tier("some-unknown-model") == "standard"

    def test_case_insensitive(self):
        assert get_model_tier("Gemini-2.5-Flash") == "quick"
        assert get_model_tier("CLAUDE-OPUS-4-6") == "thorough"

    def test_results_are_cached(self):
        tier1 = get_model_tier("gemini-2.5-pro")
        tier2 = get_model_tier("gemini-2.5-pro")
        assert tier1 == tier2 == "thorough"

    def test_empty_string_defaults_to_standard(self):
        assert get_model_tier("") == "standard"


class TestGetModelsForTier:
    def test_filters_quick_models(self):
        models = ["gemini-2.5-flash", "gemini-2.5-pro", "gpt-5.4-mini"]
        assert get_models_for_tier(models, "quick") == ["gemini-2.5-flash", "gpt-5.4-mini"]

    def test_filters_thorough_models(self):
        models = ["gemini-2.5-flash", "gemini-2.5-pro", "gpt-5.4-mini"]
        assert get_models_for_tier(models, "thorough") == ["gemini-2.5-pro"]

    def test_filters_standard_models(self):
        models = ["gemini-2.5-flash", "gpt-5.2", "gpt-5.2-codex"]
        assert get_models_for_tier(models, "standard") == ["gpt-5.2", "gpt-5.2-codex"]

    def test_empty_list_returns_empty(self):
        assert get_models_for_tier([], "quick") == []
