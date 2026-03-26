"""MCP prompt templates for nexus-mcp.

Provides discoverable workflow scaffolds for AI assistants and GUI users.
Each prompt returns a PromptResult with structured messages (assistant role
for framing, user role for the task) and a dynamic description.

Registration follows the same pattern as tools and resources: plain async
functions registered via mcp.prompt() in register_prompts().
"""

from collections.abc import Callable

from fastmcp import FastMCP

from nexus_mcp.prompts.analysis import code_review, debug, quick_triage, research, second_opinion
from nexus_mcp.prompts.comparison import compare_models
from nexus_mcp.prompts.generation import bulk_generate, implement_feature, refactor
from nexus_mcp.prompts.testing import write_tests

_ALL_PROMPTS: list[tuple[Callable[..., object], set[str]]] = [
    (code_review, {"analysis"}),
    (debug, {"analysis"}),
    (quick_triage, {"analysis"}),
    (research, {"analysis"}),
    (second_opinion, {"analysis"}),
    (implement_feature, {"generation"}),
    (refactor, {"generation"}),
    (bulk_generate, {"generation"}),
    (write_tests, {"testing"}),
    (compare_models, {"comparison"}),
]


def register_prompts(mcp: FastMCP) -> None:
    """Register all MCP prompt templates on the server.

    Called from server.py after tool and resource registration.
    """
    for prompt_fn, tags in _ALL_PROMPTS:
        mcp.prompt(tags=tags)(prompt_fn)
