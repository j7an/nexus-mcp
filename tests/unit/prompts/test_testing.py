"""Unit tests for testing prompt templates."""

from fastmcp.prompts import PromptResult

from nexus_mcp.prompts.testing import write_tests


class TestWriteTests:
    async def test_returns_prompt_result(self) -> None:
        result = await write_tests("src/parser.py")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await write_tests("src/parser.py")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await write_tests("src/parser.py")
        assert result.messages[0].role == "assistant"

    async def test_file_appears_in_user_message(self) -> None:
        result = await write_tests("src/parser.py")
        assert "src/parser.py" in result.messages[1].content.text

    async def test_framework_appears_when_provided(self) -> None:
        result = await write_tests("src/parser.py", framework="pytest")
        assert "pytest" in result.messages[1].content.text

    async def test_no_framework_line_when_empty(self) -> None:
        result = await write_tests("src/parser.py")
        assert "Framework:" not in result.messages[1].content.text

    async def test_coverage_goal_appears(self) -> None:
        result = await write_tests("src/parser.py", coverage_goal="branch")
        assert "branch" in result.messages[1].content.text

    async def test_description_includes_file(self) -> None:
        result = await write_tests("src/parser.py")
        assert "src/parser.py" in result.description

    async def test_structure_sections_present(self) -> None:
        result = await write_tests("src/parser.py")
        text = result.messages[1].content.text
        assert "Happy path" in text
        assert "Error cases" in text
        assert "Boundary conditions" in text
