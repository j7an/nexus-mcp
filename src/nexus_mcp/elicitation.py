"""ElicitationGuard — interactive parameter resolution via MCP elicitation protocol.

Wraps ctx.elicit() to interactively fill in missing parameters (CLI, model,
execution mode, prompt) before a tool call proceeds. Falls back gracefully when
the client does not support elicitation.
"""

__all__ = ["ElicitationGuard", "ResolvedParams"]

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.shared.exceptions import McpError

from nexus_mcp.config import get_runner_models
from nexus_mcp.types import PREFERENCES_KEY, ExecutionMode, SessionPreferences

logger = logging.getLogger(__name__)

_VAGUE_PROMPT_THRESHOLD = 20

type ElicitResult = (
    AcceptedElicitation[Any]
    | AcceptedElicitation[dict[str, Any]]
    | AcceptedElicitation[str]
    | AcceptedElicitation[list[str]]
    | DeclinedElicitation
    | CancelledElicitation
    | None
)


SELECTED = "user-selected"
DECLINED = "declined"


@dataclass(frozen=True)
class ResolvedParams:
    """Fully-resolved parameters after elicitation."""

    cli: str
    model: str | None
    execution_mode: ExecutionMode
    prompt_text: str
    selections: dict[str, str] = dataclasses.field(default_factory=dict)


class ElicitationGuard:
    """Guards tool calls with interactive parameter resolution.

    Wraps ctx.elicit() to interactively fill in missing parameters before
    a tool call proceeds. Caches elicitation availability at the class level
    so that a single McpError short-circuits all future elicitation attempts
    within the process lifetime.
    """

    _elicitation_available: bool | None = None

    def __init__(
        self,
        ctx: Context,
        installed_clis: list[str],
        prefs: SessionPreferences | None = None,
    ) -> None:
        self._ctx = ctx
        self._installed_clis = installed_clis
        self._prefs = prefs or SessionPreferences()

    async def _try_elicit(
        self,
        message: str,
        response_type: Any = None,
    ) -> ElicitResult:
        """Wrap ctx.elicit() with error handling and availability caching.

        Returns the elicitation result, or None if elicitation is unavailable.
        Caches unavailability at class level after first McpError.
        """
        if ElicitationGuard._elicitation_available is False:
            return None
        try:
            result = await self._ctx.elicit(message, response_type=response_type)
            ElicitationGuard._elicitation_available = True
            return result
        except McpError:
            ElicitationGuard._elicitation_available = False
            logger.debug("Elicitation not supported by client — disabling for session")
            return None

    async def _auto_suppress(self, pref_key: str) -> None:
        """Persist a suppression flag into session state."""
        raw = await self._ctx.get_state(PREFERENCES_KEY)
        current = dict(raw) if raw else {}
        current[pref_key] = False
        validated = SessionPreferences(**current)
        await self._ctx.set_state(PREFERENCES_KEY, validated.model_dump())

    # ------------------------------------------------------------------
    # Single-prompt validators
    # ------------------------------------------------------------------

    async def _validate_cli(self, cli: str | None) -> tuple[str, dict[str, str]]:
        """Resolve CLI agent via elicitation if not provided."""
        if cli is not None:
            return cli, {}
        result = await self._try_elicit(
            "Which CLI agent should handle this?",
            self._installed_clis,
        )
        if result is None or result.action != "accept":
            raise ToolError("cli is required — elicitation was declined or unavailable")
        assert isinstance(result, AcceptedElicitation)
        return str(result.data), {"cli": SELECTED}

    async def _validate_model(
        self, model: str | None, cli: str
    ) -> tuple[str | None, dict[str, str]]:
        """Resolve model via elicitation if multiple models available."""
        if model is not None:
            return model, {}
        models = get_runner_models(cli)
        if len(models) <= 1:
            return model, {}
        result = await self._try_elicit(
            f"Which {cli} model? (decline to use default)",
            list(models),
        )
        if result is not None and result.action == "accept":
            assert isinstance(result, AcceptedElicitation)
            return str(result.data), {"model": SELECTED}
        return model, {"model": DECLINED}

    async def _validate_execution_mode(
        self, mode: ExecutionMode, cli: str
    ) -> tuple[ExecutionMode, dict[str, str]]:
        """Confirm YOLO mode via elicitation, downgrade to default if declined."""
        if mode != "yolo" or self._prefs.confirm_yolo is False:
            return mode, {}
        result = await self._try_elicit(
            f"YOLO mode will auto-approve all actions in {cli}. Confirm?",
            None,
        )
        if result is None:
            return mode, {}
        if result.action == "accept":
            await self._auto_suppress("confirm_yolo")
            return mode, {"mode": SELECTED}
        logger.warning("YOLO mode declined, downgrading to default")
        return "default", {"mode": DECLINED}

    async def _validate_prompt(self, prompt_text: str) -> tuple[str, dict[str, str]]:
        """Offer elaboration for very short prompts."""
        if (
            len(prompt_text.strip()) >= _VAGUE_PROMPT_THRESHOLD
            or self._prefs.confirm_vague_prompt is False
        ):
            return prompt_text, {}
        result = await self._try_elicit(
            f"Your prompt is very short ({len(prompt_text.strip())} chars): "
            f'"{prompt_text}"\nWould you like to elaborate?',
            str,
        )
        if result is not None and result.action == "accept":
            assert isinstance(result, AcceptedElicitation)
            return str(result.data), {"prompt": SELECTED}
        return prompt_text, {"prompt": DECLINED}

    # ------------------------------------------------------------------
    # Public: single-prompt resolution
    # ------------------------------------------------------------------

    async def check_prompt(
        self,
        cli: str | None,
        model: str | None,
        execution_mode: ExecutionMode,
        prompt_text: str,
        elicit: bool = True,
    ) -> ResolvedParams:
        """Resolve all parameters, eliciting missing ones interactively."""
        if not elicit:
            if cli is None:
                raise ToolError("cli is required when elicitation is disabled")
            return ResolvedParams(
                cli=cli, model=model, execution_mode=execution_mode, prompt_text=prompt_text
            )

        resolved_cli, sel = await self._validate_cli(cli)
        selections = dict(sel)

        resolved_model, sel = await self._validate_model(model, resolved_cli)
        selections.update(sel)

        resolved_mode, sel = await self._validate_execution_mode(execution_mode, resolved_cli)
        selections.update(sel)

        resolved_prompt, sel = await self._validate_prompt(prompt_text)
        selections.update(sel)

        return ResolvedParams(
            cli=resolved_cli,
            model=resolved_model,
            execution_mode=resolved_mode,
            prompt_text=resolved_prompt,
            selections=selections,
        )

    # ------------------------------------------------------------------
    # Batch validators
    # ------------------------------------------------------------------

    async def _batch_confirm_yolo(self, tasks: list[Any]) -> list[Any]:
        """Aggregated YOLO confirmation for batch tasks."""
        yolo_indices = [i for i, t in enumerate(tasks) if t.execution_mode == "yolo"]
        if not yolo_indices or self._prefs.confirm_yolo is False:
            return tasks
        count = len(yolo_indices)
        result = await self._try_elicit(
            f"YOLO mode will auto-approve all actions for {count} task(s). Confirm?",
            None,
        )
        if result is None:
            return tasks
        if result.action == "accept":
            await self._auto_suppress("confirm_yolo")
            return tasks
        resolved = list(tasks)
        for i in yolo_indices:
            resolved[i] = resolved[i].model_copy(update={"execution_mode": "default"})
        logger.warning("YOLO mode declined for batch, downgrading %d task(s) to default", count)
        return resolved

    async def _batch_resolve_cli(self, tasks: list[Any]) -> list[Any]:
        """Aggregated CLI disambiguation for batch tasks."""
        none_cli_indices = [i for i, t in enumerate(tasks) if t.cli is None]
        if not none_cli_indices:
            return tasks
        result = await self._try_elicit(
            f"Which CLI agent should handle {len(none_cli_indices)} task(s) with no CLI set?",
            self._installed_clis,
        )
        if result is None or result.action != "accept":
            raise ToolError("cli is required — elicitation was declined or unavailable")
        assert isinstance(result, AcceptedElicitation)
        chosen_cli = str(result.data)
        resolved = list(tasks)
        for i in none_cli_indices:
            resolved[i] = resolved[i].model_copy(update={"cli": chosen_cli})
        return resolved

    async def _batch_warn_vague_prompts(self, tasks: list[Any]) -> None:
        """Advisory-only: warn about short prompts in batch. Does not modify tasks."""
        vague_indices = [
            i for i, t in enumerate(tasks) if len(t.prompt.strip()) < _VAGUE_PROMPT_THRESHOLD
        ]
        if vague_indices and self._prefs.confirm_vague_prompt is not False:
            await self._try_elicit(
                f"{len(vague_indices)} task(s) have very short prompts. Consider elaborating.",
                None,
            )

    # ------------------------------------------------------------------
    # Public: batch resolution
    # ------------------------------------------------------------------

    async def check_batch(
        self,
        tasks: list[Any],
        elicit: bool = True,
    ) -> list[Any]:
        """Resolve parameters for a batch of tasks with aggregated elicitation."""
        if not elicit:
            for task in tasks:
                if task.cli is None:
                    raise ToolError("cli is required on all tasks when elicitation is disabled")
            return tasks

        resolved = await self._batch_confirm_yolo(tasks)
        resolved = await self._batch_resolve_cli(resolved)
        await self._batch_warn_vague_prompts(resolved)

        for task in resolved:
            if task.cli is None:
                raise ToolError("cli is required on all tasks — at least one task has no CLI set")

        return resolved
