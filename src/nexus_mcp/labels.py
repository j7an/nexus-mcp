"""Label assignment for batch tasks.

Assigns unique labels to AgentTask objects, preserving explicit labels
and auto-generating from CLI names with -N suffixes for collisions.
"""

__all__ = ["assign_labels", "next_available_label"]

from nexus_mcp.types import AgentTask


def next_available_label(base: str, reserved: set[str]) -> str:
    """Return the first available label derived from base, avoiding reserved names.

    Returns base if available, otherwise base-2, base-3, etc.
    """
    if base not in reserved:
        return base
    n = 2
    while f"{base}-{n}" in reserved:
        n += 1
    return f"{base}-{n}"


def assign_labels(tasks: list[AgentTask]) -> list[AgentTask]:
    """Assign unique labels to tasks, preserving explicit ones.

    Two-pass algorithm:
    1. Reserve all explicit labels
    2. Auto-assign from cli name with -N suffixes for collisions
    """
    reserved = {t.label for t in tasks if t.label is not None}

    result: list[AgentTask] = []
    for task in tasks:
        if task.label is not None:
            result.append(task)
            continue

        label = next_available_label(task.cli or "task", reserved)
        reserved.add(label)
        result.append(task.model_copy(update={"label": label}))

    return result
