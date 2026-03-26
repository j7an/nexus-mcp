"""Analysis prompt templates: code review, debug, triage, research, second opinion."""

from fastmcp.prompts import Message, PromptResult


async def code_review(file: str, instructions: str = "") -> PromptResult:
    """Review code for issues with structured findings by severity."""
    focus = f"Focus: {instructions}\n\n" if instructions else ""
    return PromptResult(
        messages=[
            Message(
                "You are a senior code reviewer. Be specific: reference line numbers, "
                "show concrete fixes, and distinguish critical issues from suggestions.",
                role="assistant",
            ),
            Message(
                f"Review the file `{file}`.\n\n"
                f"{focus}"
                "Structure your review as:\n"
                "1. **Critical** — must fix before merge\n"
                "2. **Warnings** — should fix, risk if ignored\n"
                "3. **Suggestions** — improvements, not blocking\n\n"
                "For each finding: line number, issue, and a concrete fix.",
            ),
        ],
        description=f"Code review of {file}",
    )


async def debug(error: str, context: str = "", file: str = "") -> PromptResult:
    """Systematic diagnosis: reproduce, isolate, root cause, fix."""
    context_line = f"Context: {context}\n" if context else ""
    file_line = f"File: `{file}`\n" if file else ""
    return PromptResult(
        messages=[
            Message(
                "You are a debugger. Diagnose systematically: "
                "reproduce, isolate, identify root cause, then fix. "
                "Show your reasoning at each step.",
                role="assistant",
            ),
            Message(
                f"Error: {error}\n"
                f"{context_line}"
                f"{file_line}\n"
                "Structure your response as:\n"
                "1. **Reproduction** — how to trigger the error\n"
                "2. **Root cause** — why it happens\n"
                "3. **Fix** — concrete code change\n"
                "4. **Prevention** — how to avoid recurrence",
            ),
        ],
        description=f"Debug: {error[:80]}",
    )


async def quick_triage(description: str, file: str = "") -> PromptResult:
    """Fast assessment: what's wrong, severity, next step."""
    file_line = f"File: `{file}`\n" if file else ""
    return PromptResult(
        messages=[
            Message(
                "You are triaging an issue. Be fast and decisive. "
                "No deep analysis — just assess and recommend.",
                role="assistant",
            ),
            Message(
                f"Assess: {description}\n"
                f"{file_line}\n"
                "Answer concisely:\n"
                "1. **What's wrong** — one sentence\n"
                "2. **Severity** — critical / medium / low\n"
                "3. **Next step** — single recommended action",
            ),
        ],
        description=f"Triage: {description[:80]}",
    )


async def research(topic: str, scope: str = "focused") -> PromptResult:
    """Structured research with source citations."""
    return PromptResult(
        messages=[
            Message(
                "You are a researcher. Cite sources. "
                "Distinguish fact from inference. Be objective.",
                role="assistant",
            ),
            Message(
                f"Research: {topic}\n"
                f"Scope: {scope}\n\n"
                "Structure your response as:\n"
                "1. **Background** — context and key concepts\n"
                "2. **Current state** — what's known today\n"
                "3. **Key findings** — most important takeaways\n"
                "4. **Open questions** — what remains unclear",
            ),
        ],
        description=f"Research: {topic[:80]}",
    )


async def second_opinion(original_output: str, question: str = "Is this correct?") -> PromptResult:
    """Independent review of another AI's output."""
    return PromptResult(
        messages=[
            Message(
                "You are independently reviewing another AI's work. "
                "Do not anchor on their answer — form your own assessment first, "
                "then compare.",
                role="assistant",
            ),
            Message(
                "Another AI produced the following output:\n\n"
                "---\n"
                f"{original_output}\n"
                "---\n\n"
                f"Question: {question}\n\n"
                "Structure your response as:\n"
                "1. **Your independent assessment**\n"
                "2. **Where you agree** with the original\n"
                "3. **Where you disagree** and why\n"
                "4. **Verdict** — correct, partially correct, or incorrect",
            ),
        ],
        description=f"Second opinion: {question[:80]}",
    )
