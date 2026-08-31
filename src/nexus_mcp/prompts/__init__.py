"""Compatibility imports for MCP prompt templates."""

from nexus_mcp.mcp.prompts import register_prompts
from nexus_mcp.mcp.prompts.analysis import (
    code_review,
    debug,
    quick_triage,
    research,
    second_opinion,
)
from nexus_mcp.mcp.prompts.comparison import compare_models
from nexus_mcp.mcp.prompts.generation import bulk_generate, implement_feature, refactor
from nexus_mcp.mcp.prompts.testing import write_tests

__all__ = [
    "bulk_generate",
    "code_review",
    "compare_models",
    "debug",
    "implement_feature",
    "quick_triage",
    "refactor",
    "register_prompts",
    "research",
    "second_opinion",
    "write_tests",
]
