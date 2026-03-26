"""Testing prompt templates: write tests."""

from fastmcp.prompts import Message, PromptResult


async def write_tests(file: str, framework: str = "", coverage_goal: str = "line") -> PromptResult:
    """Generate tests for existing code with configurable coverage approach."""
    framework_line = f"Framework: {framework}\n" if framework else ""
    return PromptResult(
        messages=[
            Message(
                "You are a test engineer. Write thorough, maintainable tests. "
                "Follow the project's existing test patterns and conventions.",
                role="assistant",
            ),
            Message(
                f"Write tests for `{file}`.\n"
                f"{framework_line}"
                f"Coverage goal: {coverage_goal}\n\n"
                "Structure your tests as:\n"
                "1. **Happy path** — expected behavior works correctly\n"
                "2. **Error cases** — invalid inputs, failures, edge cases\n"
                "3. **Boundary conditions** — limits, empty inputs, large inputs",
            ),
        ],
        description=f"Write tests for {file}",
    )
