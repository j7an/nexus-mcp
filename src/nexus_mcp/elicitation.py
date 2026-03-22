"""ElicitationGuard — interactive parameter resolution via MCP elicitation protocol.

Wraps ctx.elicit() to interactively fill in missing parameters (CLI, model,
execution mode, prompt) before a tool call proceeds. Falls back gracefully when
the client does not support elicitation.
"""

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
from nexus_mcp.types import PREFERENCES_KEY, SessionPreferences

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


@dataclass(frozen=True)
class ResolvedParams:
    """Fully-resolved parameters after elicitation."""

    cli: str
    model: str | None
    execution_mode: str
    prompt_text: str


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

    async def check_prompt(
        self,
        cli: str | None,
        model: str | None,
        execution_mode: str,
        prompt_text: str,
        elicit: bool = True,
    ) -> ResolvedParams:
        """Resolve all parameters, eliciting missing ones interactively.

        Args:
            cli: CLI agent name, or None to trigger disambiguation.
            model: Model name, or None to trigger selection.
            execution_mode: Execution mode ("default" or "yolo").
            prompt_text: The prompt text; short prompts trigger elaboration.
            elicit: If False, skip all elicitation and raise on missing params.

        Returns:
            ResolvedParams with all fields populated.

        Raises:
            ToolError: If a required parameter is missing and elicitation is
                unavailable, declined, or disabled.
        """
        if not elicit:
            if cli is None:
                raise ToolError("cli is required when elicitation is disabled")
            return ResolvedParams(
                cli=cli,
                model=model,
                execution_mode=execution_mode,
                prompt_text=prompt_text,
            )

        resolved_cli = cli
        resolved_model = model
        resolved_mode = execution_mode
        resolved_prompt = prompt_text

        # Trigger #1: CLI disambiguation
        if resolved_cli is None:
            result = await self._try_elicit(
                "Which CLI agent should handle this?",
                self._installed_clis,
            )
            if result is None or result.action != "accept":
                raise ToolError("cli is required — elicitation was declined or unavailable")
            assert isinstance(result, AcceptedElicitation)
            resolved_cli = str(result.data)

        # Trigger #2: Model selection
        if resolved_model is None and resolved_cli:
            models = get_runner_models(resolved_cli)
            if len(models) > 1:
                result = await self._try_elicit(
                    f"Which {resolved_cli} model? (decline to use default)",
                    list(models),
                )
                if result is not None and result.action == "accept":
                    assert isinstance(result, AcceptedElicitation)
                    resolved_model = str(result.data)

        # Trigger #3: YOLO confirmation
        if resolved_mode == "yolo" and self._prefs.confirm_yolo is not False:
            result = await self._try_elicit(
                f"YOLO mode will auto-approve all actions in {resolved_cli}. Confirm?",
                None,
            )
            if result is None:
                pass  # elicitation unavailable, proceed
            elif result.action == "accept":
                await self._auto_suppress("confirm_yolo")
            else:
                resolved_mode = "default"
                logger.warning("YOLO mode declined, downgrading to default")

        # Trigger #4: Vague prompt check (no auto-suppress)
        if (
            len(resolved_prompt.strip()) < _VAGUE_PROMPT_THRESHOLD
            and self._prefs.confirm_vague_prompt is not False
        ):
            result = await self._try_elicit(
                f"Your prompt is very short ({len(resolved_prompt.strip())} chars): "
                f'"{resolved_prompt}"\nWould you like to elaborate?',
                str,
            )
            if result is not None and result.action == "accept":
                assert isinstance(result, AcceptedElicitation)
                resolved_prompt = str(result.data)

        # resolved_cli is guaranteed non-None here: either it was provided,
        # or Trigger #1 set it (raising ToolError if it couldn't).
        assert resolved_cli is not None
        return ResolvedParams(
            cli=resolved_cli,
            model=resolved_model,
            execution_mode=resolved_mode,
            prompt_text=resolved_prompt,
        )

    async def check_batch(
        self,
        tasks: list[Any],
        elicit: bool = True,
    ) -> list[Any]:
        """Resolve parameters for a batch of tasks.

        Stub — implemented in Task 8.
        """
        return tasks
