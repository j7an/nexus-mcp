"""Unit tests for generation prompt templates."""

from fastmcp.prompts import PromptResult

from nexus_mcp.prompts.generation import bulk_generate, implement_feature, refactor


class TestImplementFeature:
    async def test_returns_prompt_result(self) -> None:
        result = await implement_feature("add login endpoint")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await implement_feature("add login endpoint")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await implement_feature("add login endpoint")
        assert result.messages[0].role == "assistant"

    async def test_description_appears_in_user_message(self) -> None:
        result = await implement_feature("add login endpoint")
        assert "add login endpoint" in result.messages[1].content.text

    async def test_language_appears_when_provided(self) -> None:
        result = await implement_feature("add login endpoint", language="Python")
        assert "Python" in result.messages[1].content.text

    async def test_no_language_line_when_empty(self) -> None:
        result = await implement_feature("add login endpoint")
        assert "Language:" not in result.messages[1].content.text

    async def test_constraints_appear_when_provided(self) -> None:
        result = await implement_feature("add login endpoint", constraints="no third-party libs")
        assert "no third-party libs" in result.messages[1].content.text

    async def test_no_constraints_line_when_empty(self) -> None:
        result = await implement_feature("add login endpoint")
        assert "Constraints:" not in result.messages[1].content.text

    async def test_description_includes_description(self) -> None:
        result = await implement_feature("add login endpoint")
        assert "add login endpoint" in result.description

    async def test_structure_sections_present(self) -> None:
        result = await implement_feature("add login endpoint")
        text = result.messages[1].content.text
        assert "Implementation" in text
        assert "Tests" in text
        assert "Edge cases" in text


class TestRefactor:
    async def test_returns_prompt_result(self) -> None:
        result = await refactor("src/utils.py", "reduce complexity")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await refactor("src/utils.py", "reduce complexity")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await refactor("src/utils.py", "reduce complexity")
        assert result.messages[0].role == "assistant"

    async def test_file_and_goal_appear_in_user_message(self) -> None:
        result = await refactor("src/utils.py", "reduce complexity")
        text = result.messages[1].content.text
        assert "src/utils.py" in text
        assert "reduce complexity" in text

    async def test_constraints_appear_when_provided(self) -> None:
        result = await refactor("src/utils.py", "reduce complexity", constraints="keep API stable")
        assert "keep API stable" in result.messages[1].content.text

    async def test_no_constraints_line_when_empty(self) -> None:
        result = await refactor("src/utils.py", "reduce complexity")
        assert "Constraints:" not in result.messages[1].content.text

    async def test_description_includes_file_and_goal(self) -> None:
        result = await refactor("src/utils.py", "reduce complexity")
        assert "src/utils.py" in result.description
        assert "reduce complexity" in result.description

    async def test_structure_sections_present(self) -> None:
        result = await refactor("src/utils.py", "reduce complexity")
        text = result.messages[1].content.text
        assert "Before" in text
        assert "Changes" in text
        assert "After" in text
        assert "Verification" in text


class TestBulkGenerate:
    async def test_returns_prompt_result(self) -> None:
        result = await bulk_generate("Write a greeting for {name}")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await bulk_generate("Write a greeting for {name}")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await bulk_generate("Write a greeting for {name}")
        assert result.messages[0].role == "assistant"

    async def test_template_appears_in_user_message(self) -> None:
        result = await bulk_generate("Write a greeting for {name}")
        assert "Write a greeting for {name}" in result.messages[1].content.text

    async def test_variables_appear_in_user_message(self) -> None:
        result = await bulk_generate("Write a greeting for {name}", variables=[{"name": "Alice"}])
        assert "Alice" in result.messages[1].content.text

    async def test_none_variables_accepted(self) -> None:
        result = await bulk_generate("Write a greeting for {name}", variables=None)
        assert isinstance(result, PromptResult)
        assert "[]" in result.messages[1].content.text

    async def test_empty_variables_accepted(self) -> None:
        result = await bulk_generate("Write a greeting for {name}", variables=[])
        assert isinstance(result, PromptResult)
        assert "[]" in result.messages[1].content.text

    async def test_description_includes_template(self) -> None:
        result = await bulk_generate("Write a greeting for {name}")
        assert "Write a greeting for {name}" in result.description
