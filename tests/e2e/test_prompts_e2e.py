"""E2E MCP prompt tests using FastMCP's in-process Client.

Tests the full MCP stack for prompts:
- Prompt discovery via list_prompts() → JSON-RPC
- Prompt rendering via get_prompt() → JSON-RPC
- Parameter schema validation

Mock boundary: CLI detection only (via autouse mock_cli_detection).
All layers above run for real, including JSON-RPC dispatch.
"""

import pytest

EXPECTED_PROMPTS = {
    "code_review",
    "debug",
    "quick_triage",
    "research",
    "second_opinion",
    "implement_feature",
    "refactor",
    "bulk_generate",
    "write_tests",
    "compare_models",
}


@pytest.mark.e2e
class TestPromptDiscovery:
    """Verify MCP prompt registration via list_prompts() JSON-RPC call."""

    async def test_list_prompts_returns_all_ten(self, mcp_client):
        """list_prompts() returns exactly 10 registered prompts."""
        prompts = await mcp_client.list_prompts()
        names = {p.name for p in prompts}
        assert names == EXPECTED_PROMPTS

    async def test_all_prompts_have_descriptions(self, mcp_client):
        """Every prompt has a non-empty description (from docstring)."""
        prompts = await mcp_client.list_prompts()
        for p in prompts:
            assert p.description, f"Prompt {p.name!r} missing description"

    async def test_all_prompts_have_arguments(self, mcp_client):
        """Every prompt has at least one argument."""
        prompts = await mcp_client.list_prompts()
        for p in prompts:
            assert p.arguments, f"Prompt {p.name!r} has no arguments"


@pytest.mark.e2e
class TestPromptRendering:
    """Verify get_prompt() returns valid PromptResult via JSON-RPC."""

    async def test_code_review_renders(self, mcp_client):
        result = await mcp_client.get_prompt("code_review", arguments={"file": "app.py"})
        assert len(result.messages) == 2
        assert result.messages[0].role == "assistant"
        assert result.messages[1].role == "user"

    async def test_code_review_with_instructions(self, mcp_client):
        result = await mcp_client.get_prompt(
            "code_review",
            arguments={"file": "app.py", "instructions": "security"},
        )
        assert "security" in result.messages[1].content.text

    async def test_debug_renders(self, mcp_client):
        result = await mcp_client.get_prompt("debug", arguments={"error": "TypeError: None"})
        assert len(result.messages) == 2
        assert "TypeError" in result.messages[1].content.text

    async def test_quick_triage_renders(self, mcp_client):
        result = await mcp_client.get_prompt("quick_triage", arguments={"description": "API 500s"})
        assert len(result.messages) == 2

    async def test_research_renders(self, mcp_client):
        result = await mcp_client.get_prompt("research", arguments={"topic": "WebSockets"})
        assert len(result.messages) == 2
        assert "WebSockets" in result.messages[1].content.text

    async def test_second_opinion_renders(self, mcp_client):
        result = await mcp_client.get_prompt(
            "second_opinion", arguments={"original_output": "The fix is X"}
        )
        assert len(result.messages) == 2
        assert "The fix is X" in result.messages[1].content.text

    async def test_implement_feature_renders(self, mcp_client):
        result = await mcp_client.get_prompt(
            "implement_feature", arguments={"description": "Add auth"}
        )
        assert len(result.messages) == 2
        assert "Add auth" in result.messages[1].content.text

    async def test_refactor_renders(self, mcp_client):
        result = await mcp_client.get_prompt(
            "refactor", arguments={"file": "app.py", "goal": "extract class"}
        )
        assert len(result.messages) == 2

    async def test_write_tests_renders(self, mcp_client):
        result = await mcp_client.get_prompt("write_tests", arguments={"file": "auth.py"})
        assert len(result.messages) == 2

    async def test_compare_models_renders(self, mcp_client):
        result = await mcp_client.get_prompt("compare_models", arguments={"prompt": "Explain GIL"})
        assert len(result.messages) == 2
        assert "Explain GIL" in result.messages[1].content.text

    async def test_bulk_generate_renders(self, mcp_client):
        result = await mcp_client.get_prompt(
            "bulk_generate",
            arguments={"template": "Docs for {fn}", "variables": '[{"fn": "foo"}]'},
        )
        assert len(result.messages) == 2
