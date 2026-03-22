"""Tests for ElicitationGuard — interactive parameter resolution."""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.server.elicitation import (
    AcceptedElicitation,
    CancelledElicitation,
    DeclinedElicitation,
)
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from nexus_mcp.elicitation import ElicitationGuard
from nexus_mcp.types import SessionPreferences

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx() -> AsyncMock:
    ctx = AsyncMock(spec=Context)
    ctx.elicit = AsyncMock()
    ctx.get_state = AsyncMock(return_value=None)
    ctx.set_state = AsyncMock()
    return ctx


@pytest.fixture
def installed_clis() -> list[str]:
    return ["gemini", "codex", "claude"]


# ---------------------------------------------------------------------------
# Task 3: TestShortCircuit
# ---------------------------------------------------------------------------


class TestShortCircuit:
    @pytest.fixture(autouse=True)
    def _reset_class_cache(self) -> None:
        ElicitationGuard._elicitation_available = None
        yield
        ElicitationGuard._elicitation_available = None

    async def test_elicit_false_skips_all_checks(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="default",
            prompt_text="explain quantum computing in depth",
            elicit=False,
        )
        mock_ctx.elicit.assert_not_called()
        assert result.cli == "gemini"

    async def test_unsupported_client_skips_silently(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.side_effect = McpError(ErrorData(code=-32600, message="not supported"))
        guard = ElicitationGuard(mock_ctx, installed_clis)
        with pytest.raises(ToolError):
            await guard.check_prompt(
                cli=None,
                model=None,
                execution_mode="default",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )
        assert ElicitationGuard._elicitation_available is False

    async def test_no_triggers_fire_when_all_params_provided(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_prompt(
            cli="gemini",
            model="gemini-2.5-flash",
            execution_mode="default",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.elicit.assert_not_called()
        assert result.cli == "gemini"
        assert result.model == "gemini-2.5-flash"
        assert result.execution_mode == "default"


# ---------------------------------------------------------------------------
# Task 4: TestCliDisambiguation
# ---------------------------------------------------------------------------


class TestCliDisambiguation:
    @pytest.fixture(autouse=True)
    def _reset_class_cache(self) -> None:
        ElicitationGuard._elicitation_available = None
        yield
        ElicitationGuard._elicitation_available = None

    async def test_fires_when_cli_none(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = AcceptedElicitation(data="gemini")
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_prompt(
            cli=None,
            model=None,
            execution_mode="default",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.elicit.assert_called_once()
        assert result.cli == "gemini"

    async def test_decline_raises_tool_error(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = DeclinedElicitation()
        guard = ElicitationGuard(mock_ctx, installed_clis)
        with pytest.raises(ToolError):
            await guard.check_prompt(
                cli=None,
                model=None,
                execution_mode="default",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )

    async def test_cancel_raises_tool_error(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = CancelledElicitation()
        guard = ElicitationGuard(mock_ctx, installed_clis)
        with pytest.raises(ToolError):
            await guard.check_prompt(
                cli=None,
                model=None,
                execution_mode="default",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )

    async def test_does_not_fire_when_cli_provided(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="default",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.elicit.assert_not_called()
        assert result.cli == "gemini"


# ---------------------------------------------------------------------------
# Task 5: TestModelSelection
# ---------------------------------------------------------------------------


class TestModelSelection:
    @pytest.fixture(autouse=True)
    def _reset_class_cache(self) -> None:
        ElicitationGuard._elicitation_available = None
        yield
        ElicitationGuard._elicitation_available = None

    async def test_fires_when_model_none_and_multiple_available(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = AcceptedElicitation(data="pro")
        with patch("nexus_mcp.elicitation.get_runner_models", return_value=("pro", "flash")):
            guard = ElicitationGuard(mock_ctx, installed_clis)
            result = await guard.check_prompt(
                cli="gemini",
                model=None,
                execution_mode="default",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )
        mock_ctx.elicit.assert_called_once()
        assert result.model == "pro"

    async def test_decline_uses_default_model(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = DeclinedElicitation()
        with patch("nexus_mcp.elicitation.get_runner_models", return_value=("pro", "flash")):
            guard = ElicitationGuard(mock_ctx, installed_clis)
            result = await guard.check_prompt(
                cli="gemini",
                model=None,
                execution_mode="default",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )
        assert result.model is None

    async def test_does_not_fire_when_model_provided(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        with patch("nexus_mcp.elicitation.get_runner_models", return_value=("pro", "flash")):
            guard = ElicitationGuard(mock_ctx, installed_clis)
            result = await guard.check_prompt(
                cli="gemini",
                model="flash",
                execution_mode="default",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )
        mock_ctx.elicit.assert_not_called()
        assert result.model == "flash"

    async def test_does_not_fire_when_single_model(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        with patch("nexus_mcp.elicitation.get_runner_models", return_value=("flash",)):
            guard = ElicitationGuard(mock_ctx, installed_clis)
            await guard.check_prompt(
                cli="gemini",
                model=None,
                execution_mode="default",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )
        mock_ctx.elicit.assert_not_called()

    async def test_does_not_fire_when_no_models(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        with patch("nexus_mcp.elicitation.get_runner_models", return_value=()):
            guard = ElicitationGuard(mock_ctx, installed_clis)
            await guard.check_prompt(
                cli="gemini",
                model=None,
                execution_mode="default",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )
        mock_ctx.elicit.assert_not_called()


# ---------------------------------------------------------------------------
# Task 6: TestYoloConfirmation
# ---------------------------------------------------------------------------


class TestYoloConfirmation:
    @pytest.fixture(autouse=True)
    def _reset_class_cache(self) -> None:
        ElicitationGuard._elicitation_available = None
        yield
        ElicitationGuard._elicitation_available = None

    async def test_fires_when_yolo(self, mock_ctx: AsyncMock, installed_clis: list[str]) -> None:
        mock_ctx.elicit.return_value = AcceptedElicitation(data={})
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="yolo",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.elicit.assert_called_once()
        assert result.execution_mode == "yolo"

    async def test_decline_downgrades_to_default(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = DeclinedElicitation()
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="yolo",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        assert result.execution_mode == "default"

    async def test_does_not_fire_when_default_mode(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        guard = ElicitationGuard(mock_ctx, installed_clis)
        await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="default",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.elicit.assert_not_called()

    async def test_auto_suppresses_after_accept(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = AcceptedElicitation(data={})
        mock_ctx.get_state.return_value = None
        guard = ElicitationGuard(mock_ctx, installed_clis)
        await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="yolo",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.set_state.assert_called_once()
        call_args = mock_ctx.set_state.call_args
        assert call_args[0][0] == "nexus:preferences"
        saved = call_args[0][1]
        assert saved["confirm_yolo"] is False

    async def test_suppressed_yolo_skips_elicitation(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        prefs = SessionPreferences(confirm_yolo=False)
        guard = ElicitationGuard(mock_ctx, installed_clis, prefs=prefs)
        result = await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="yolo",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.elicit.assert_not_called()
        assert result.execution_mode == "yolo"

    async def test_reset_suppression_re_enables_prompt(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = AcceptedElicitation(data={})
        prefs = SessionPreferences(confirm_yolo=None)
        guard = ElicitationGuard(mock_ctx, installed_clis, prefs=prefs)
        await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="yolo",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.elicit.assert_called_once()


# ---------------------------------------------------------------------------
# Task 7: TestVaguePromptCheck
# ---------------------------------------------------------------------------


class TestVaguePromptCheck:
    @pytest.fixture(autouse=True)
    def _reset_class_cache(self) -> None:
        ElicitationGuard._elicitation_available = None
        yield
        ElicitationGuard._elicitation_available = None

    async def test_fires_under_threshold(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = AcceptedElicitation(data="elaborated prompt text")
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="default",
            prompt_text="explain QC",
            elicit=True,
        )
        mock_ctx.elicit.assert_called_once()
        assert result.prompt_text == "elaborated prompt text"

    async def test_decline_keeps_original(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = DeclinedElicitation()
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="default",
            prompt_text="explain QC",
            elicit=True,
        )
        assert result.prompt_text == "explain QC"

    async def test_does_not_fire_above_threshold(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        guard = ElicitationGuard(mock_ctx, installed_clis)
        await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="default",
            prompt_text="explain quantum computing in depth",
            elicit=True,
        )
        mock_ctx.elicit.assert_not_called()

    async def test_does_not_auto_suppress(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        mock_ctx.elicit.return_value = AcceptedElicitation(data="elaborated prompt text")
        guard = ElicitationGuard(mock_ctx, installed_clis)
        await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="default",
            prompt_text="explain QC",
            elicit=True,
        )
        mock_ctx.set_state.assert_not_called()

    async def test_explicit_suppress_skips_trigger(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        prefs = SessionPreferences(confirm_vague_prompt=False)
        guard = ElicitationGuard(mock_ctx, installed_clis, prefs=prefs)
        result = await guard.check_prompt(
            cli="gemini",
            model=None,
            execution_mode="default",
            prompt_text="explain QC",
            elicit=True,
        )
        mock_ctx.elicit.assert_not_called()
        assert result.prompt_text == "explain QC"
