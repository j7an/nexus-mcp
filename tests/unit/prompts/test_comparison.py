"""Unit tests for comparison prompt templates."""

from fastmcp.prompts import PromptResult

from nexus_mcp.prompts.comparison import compare_models


class TestCompareModels:
    async def test_returns_prompt_result(self) -> None:
        result = await compare_models("Explain recursion")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await compare_models("Explain recursion")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await compare_models("Explain recursion")
        assert result.messages[0].role == "assistant"

    async def test_prompt_appears_in_user_message(self) -> None:
        result = await compare_models("Explain recursion")
        assert "Explain recursion" in result.messages[1].content.text

    async def test_default_criteria_is_quality(self) -> None:
        result = await compare_models("Explain recursion")
        assert "quality" in result.messages[1].content.text

    async def test_custom_criteria_appears(self) -> None:
        result = await compare_models("Explain recursion", criteria="accuracy and depth")
        assert "accuracy and depth" in result.messages[1].content.text

    async def test_description_includes_prompt(self) -> None:
        result = await compare_models("Explain recursion")
        assert "Explain recursion" in result.description

    async def test_structure_sections_present(self) -> None:
        result = await compare_models("Explain recursion")
        text = result.messages[1].content.text
        assert "Per-model assessment" in text
        assert "Ranking" in text
        assert "Recommendation" in text
