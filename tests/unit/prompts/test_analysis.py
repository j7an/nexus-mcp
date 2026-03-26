"""Unit tests for analysis prompt templates."""

from fastmcp.prompts import PromptResult

from nexus_mcp.prompts.analysis import code_review, debug, quick_triage, research, second_opinion


class TestCodeReview:
    async def test_returns_prompt_result(self) -> None:
        result = await code_review("src/foo.py")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await code_review("src/foo.py")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await code_review("src/foo.py")
        assert result.messages[0].role == "assistant"

    async def test_second_message_is_user(self) -> None:
        result = await code_review("src/foo.py")
        assert result.messages[1].role == "user"

    async def test_file_appears_in_user_message(self) -> None:
        result = await code_review("src/foo.py")
        assert "src/foo.py" in result.messages[1].content.text

    async def test_description_includes_file(self) -> None:
        result = await code_review("src/foo.py")
        assert "src/foo.py" in result.description

    async def test_instructions_appear_when_provided(self) -> None:
        result = await code_review("src/foo.py", instructions="focus on security")
        assert "focus on security" in result.messages[1].content.text

    async def test_no_focus_line_when_no_instructions(self) -> None:
        result = await code_review("src/foo.py")
        assert "Focus:" not in result.messages[1].content.text

    async def test_structure_sections_present(self) -> None:
        result = await code_review("src/foo.py")
        text = result.messages[1].content.text
        assert "Critical" in text
        assert "Warnings" in text
        assert "Suggestions" in text


class TestDebug:
    async def test_returns_prompt_result(self) -> None:
        result = await debug("TypeError: NoneType")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await debug("TypeError: NoneType")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await debug("TypeError: NoneType")
        assert result.messages[0].role == "assistant"

    async def test_error_appears_in_user_message(self) -> None:
        result = await debug("TypeError: NoneType")
        assert "TypeError: NoneType" in result.messages[1].content.text

    async def test_context_appears_when_provided(self) -> None:
        result = await debug("TypeError: NoneType", context="line 42")
        assert "line 42" in result.messages[1].content.text

    async def test_file_appears_when_provided(self) -> None:
        result = await debug("TypeError: NoneType", file="app.py")
        assert "app.py" in result.messages[1].content.text

    async def test_no_context_line_when_empty(self) -> None:
        result = await debug("TypeError: NoneType")
        assert "Context:" not in result.messages[1].content.text

    async def test_no_file_line_when_empty(self) -> None:
        result = await debug("TypeError: NoneType")
        assert "File:" not in result.messages[1].content.text

    async def test_description_includes_error(self) -> None:
        result = await debug("TypeError: NoneType")
        assert "TypeError: NoneType" in result.description

    async def test_description_truncates_long_error(self) -> None:
        long_error = "E" * 100
        result = await debug(long_error)
        assert len(result.description) < len(long_error) + 20

    async def test_structure_sections_present(self) -> None:
        result = await debug("TypeError: NoneType")
        text = result.messages[1].content.text
        assert "Reproduction" in text
        assert "Root cause" in text
        assert "Fix" in text
        assert "Prevention" in text


class TestQuickTriage:
    async def test_returns_prompt_result(self) -> None:
        result = await quick_triage("server is down")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await quick_triage("server is down")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await quick_triage("server is down")
        assert result.messages[0].role == "assistant"

    async def test_description_appears_in_user_message(self) -> None:
        result = await quick_triage("server is down")
        assert "server is down" in result.messages[1].content.text

    async def test_file_appears_when_provided(self) -> None:
        result = await quick_triage("server is down", file="server.py")
        assert "server.py" in result.messages[1].content.text

    async def test_no_file_line_when_empty(self) -> None:
        result = await quick_triage("server is down")
        assert "File:" not in result.messages[1].content.text

    async def test_description_includes_text(self) -> None:
        result = await quick_triage("server is down")
        assert "server is down" in result.description

    async def test_structure_sections_present(self) -> None:
        result = await quick_triage("server is down")
        text = result.messages[1].content.text
        assert "What's wrong" in text
        assert "Severity" in text
        assert "Next step" in text


class TestResearch:
    async def test_returns_prompt_result(self) -> None:
        result = await research("quantum computing")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await research("quantum computing")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await research("quantum computing")
        assert result.messages[0].role == "assistant"

    async def test_topic_appears_in_user_message(self) -> None:
        result = await research("quantum computing")
        assert "quantum computing" in result.messages[1].content.text

    async def test_default_scope_is_focused(self) -> None:
        result = await research("quantum computing")
        assert "focused" in result.messages[1].content.text

    async def test_custom_scope_appears(self) -> None:
        result = await research("quantum computing", scope="broad")
        assert "broad" in result.messages[1].content.text

    async def test_description_includes_topic(self) -> None:
        result = await research("quantum computing")
        assert "quantum computing" in result.description

    async def test_structure_sections_present(self) -> None:
        result = await research("quantum computing")
        text = result.messages[1].content.text
        assert "Background" in text
        assert "Current state" in text
        assert "Key findings" in text
        assert "Open questions" in text


class TestSecondOpinion:
    async def test_returns_prompt_result(self) -> None:
        result = await second_opinion("The answer is 42.")
        assert isinstance(result, PromptResult)

    async def test_has_two_messages(self) -> None:
        result = await second_opinion("The answer is 42.")
        assert len(result.messages) == 2

    async def test_first_message_is_assistant(self) -> None:
        result = await second_opinion("The answer is 42.")
        assert result.messages[0].role == "assistant"

    async def test_original_output_appears_in_user_message(self) -> None:
        result = await second_opinion("The answer is 42.")
        assert "The answer is 42." in result.messages[1].content.text

    async def test_default_question(self) -> None:
        result = await second_opinion("The answer is 42.")
        assert "Is this correct?" in result.messages[1].content.text

    async def test_custom_question_appears(self) -> None:
        result = await second_opinion("The answer is 42.", question="Is the math right?")
        assert "Is the math right?" in result.messages[1].content.text

    async def test_description_includes_question(self) -> None:
        result = await second_opinion("The answer is 42.", question="Is the math right?")
        assert "Is the math right?" in result.description

    async def test_structure_sections_present(self) -> None:
        result = await second_opinion("The answer is 42.")
        text = result.messages[1].content.text
        assert "independent assessment" in text
        assert "agree" in text
        assert "disagree" in text
        assert "Verdict" in text
