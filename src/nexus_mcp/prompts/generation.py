"""Generation prompt templates: implement feature, refactor, bulk generate."""

import json

from fastmcp.prompts import Message, PromptResult


async def implement_feature(
    description: str, language: str = "", constraints: str = ""
) -> PromptResult:
    """Generate feature implementation with quality checklist."""
    language_line = f"Language: {language}\n" if language else ""
    constraints_line = f"Constraints: {constraints}\n" if constraints else ""
    return PromptResult(
        messages=[
            Message(
                "You are a software engineer implementing a feature. "
                "Write clean, tested, production-quality code. "
                "Follow the project's existing patterns and conventions.",
                role="assistant",
            ),
            Message(
                f"Implement: {description}\n"
                f"{language_line}"
                f"{constraints_line}\n"
                "Deliverables:\n"
                "1. **Implementation** — working code\n"
                "2. **Tests** — cover happy path and error cases\n"
                "3. **Edge cases** — document any assumptions or limitations",
            ),
        ],
        description=f"Implement: {description[:80]}",
    )


async def refactor(file: str, goal: str, constraints: str = "") -> PromptResult:
    """Restructure code while preserving behavior."""
    constraints_line = f"Constraints: {constraints}\n" if constraints else ""
    return PromptResult(
        messages=[
            Message(
                "You are refactoring code. Preserve all existing behavior. "
                "No feature changes. Verify before and after.",
                role="assistant",
            ),
            Message(
                f"Refactor `{file}`.\n"
                f"Goal: {goal}\n"
                f"{constraints_line}\n"
                "Structure your response as:\n"
                "1. **Before** — current state and why it needs change\n"
                "2. **Changes** — what you changed and why\n"
                "3. **After** — the refactored code\n"
                "4. **Verification** — how to confirm behavior is preserved",
            ),
        ],
        description=f"Refactor {file}: {goal[:60]}",
    )


async def bulk_generate(
    template: str, variables: list[dict[str, object]] | None = None
) -> PromptResult:
    """Expand a template across variable sets for batch generation."""
    variables = variables or []
    variables_block = json.dumps(variables, indent=2) if variables else "[]"
    return PromptResult(
        messages=[
            Message(
                "You are generating content from a template applied to multiple inputs. "
                "Produce consistent, high-quality output for each variable set.",
                role="assistant",
            ),
            Message(
                f"Template: {template}\n\n"
                f"Variables:\n```json\n{variables_block}\n```\n\n"
                "For each variable set, expand the template and produce the output. "
                "Maintain consistent quality and format across all expansions.",
            ),
        ],
        description=f"Bulk generate: {template[:80]}",
    )
