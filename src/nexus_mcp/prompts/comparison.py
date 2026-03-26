"""Comparison prompt templates: multi-model comparison."""

from fastmcp.prompts import Message, PromptResult


async def compare_models(prompt: str, criteria: str = "quality") -> PromptResult:
    """Compare outputs from multiple AI models."""
    return PromptResult(
        messages=[
            Message(
                "You are evaluating AI model responses. Compare the outputs objectively. "
                "Do not favor any model — assess each on its merits.",
                role="assistant",
            ),
            Message(
                "The following prompt was sent to multiple AI models:\n\n"
                f"> {prompt}\n\n"
                f"Evaluate each response on: {criteria}.\n"
                "Rank them and explain your reasoning.\n\n"
                "Structure your comparison as:\n"
                "1. **Per-model assessment** — strengths and weaknesses of each\n"
                "2. **Ranking** — ordered list with reasoning\n"
                "3. **Recommendation** — which to use and when",
            ),
        ],
        description=f"Multi-model comparison: {prompt[:80]}",
    )
