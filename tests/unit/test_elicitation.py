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
from nexus_mcp.types import AgentTask, SessionPreferences

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx() -> AsyncMock:
    ctx = AsyncMock(spec=Context)
    ctx.elicit = AsyncMock()
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
        with (
            patch("nexus_mcp.elicitation.load_preferences", return_value=None),
            patch("nexus_mcp.elicitation.save_preferences"),
        ):
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
        with (
            patch("nexus_mcp.elicitation.load_preferences", return_value=None) as mock_load,
            patch("nexus_mcp.elicitation.save_preferences") as mock_save,
        ):
            guard = ElicitationGuard(mock_ctx, installed_clis)
            await guard.check_prompt(
                cli="gemini",
                model=None,
                execution_mode="yolo",
                prompt_text="explain quantum computing in depth",
                elicit=True,
            )
            mock_load.assert_awaited_once()
            mock_save.assert_awaited_once()
            saved = mock_save.call_args.args[1]
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
        with (
            patch("nexus_mcp.elicitation.load_preferences", return_value=None),
            patch("nexus_mcp.elicitation.save_preferences"),
        ):
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
        with patch("nexus_mcp.elicitation.save_preferences") as mock_save:
            guard = ElicitationGuard(mock_ctx, installed_clis)
            await guard.check_prompt(
                cli="gemini",
                model=None,
                execution_mode="default",
                prompt_text="explain QC",
                elicit=True,
            )
            mock_save.assert_not_awaited()

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


# ---------------------------------------------------------------------------
# Task 8: TestBatchElicitation
# ---------------------------------------------------------------------------


class TestBatchElicitation:
    @pytest.fixture(autouse=True)
    def _reset_class_cache(self) -> None:
        ElicitationGuard._elicitation_available = None
        yield
        ElicitationGuard._elicitation_available = None

    async def test_aggregates_yolo_confirmation(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        """3 out of 5 YOLO tasks triggers a single elicit with count in message."""
        mock_ctx.elicit.return_value = AcceptedElicitation(data={})
        tasks = [
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="yolo"),
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="yolo"),
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="yolo"),
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="default"),
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="default"),
        ]
        with (
            patch("nexus_mcp.elicitation.load_preferences", return_value=None),
            patch("nexus_mcp.elicitation.save_preferences"),
        ):
            guard = ElicitationGuard(mock_ctx, installed_clis)
            await guard.check_batch(tasks, elicit=True)
        mock_ctx.elicit.assert_called_once()
        call_message = mock_ctx.elicit.call_args.args[0]
        assert "3" in call_message

    async def test_batch_decline_yolo_downgrades_all(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        """Declining YOLO confirmation downgrades all YOLO tasks to default."""
        mock_ctx.elicit.return_value = DeclinedElicitation()
        tasks = [
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="yolo"),
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="yolo"),
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="default"),
        ]
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_batch(tasks, elicit=True)
        assert result[0].execution_mode == "default"
        assert result[1].execution_mode == "default"
        assert result[2].execution_mode == "default"

    async def test_batch_validates_cli_required(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        """elicit=False + task with cli=None → ToolError."""
        tasks = [
            AgentTask(cli="gemini", prompt="do something useful here"),
            AgentTask(cli=None, prompt="do something useful here"),
        ]
        guard = ElicitationGuard(mock_ctx, installed_clis)
        with pytest.raises(ToolError):
            await guard.check_batch(tasks, elicit=False)
        mock_ctx.elicit.assert_not_called()

    async def test_batch_aggregates_cli_disambiguation(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        """2 tasks with cli=None → single elicit call, both tasks get chosen CLI."""
        mock_ctx.elicit.return_value = AcceptedElicitation(data="gemini")
        tasks = [
            AgentTask(cli=None, prompt="do something useful here"),
            AgentTask(cli=None, prompt="do something useful here"),
            AgentTask(cli="codex", prompt="do something useful here"),
        ]
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_batch(tasks, elicit=True)
        mock_ctx.elicit.assert_called_once()
        assert result[0].cli == "gemini"
        assert result[1].cli == "gemini"
        assert result[2].cli == "codex"

    async def test_batch_cli_decline_raises(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        """Declining CLI disambiguation raises ToolError."""
        mock_ctx.elicit.return_value = DeclinedElicitation()
        tasks = [
            AgentTask(cli=None, prompt="do something useful here"),
        ]
        guard = ElicitationGuard(mock_ctx, installed_clis)
        with pytest.raises(ToolError):
            await guard.check_batch(tasks, elicit=True)

    async def test_batch_elicit_false_skips(
        self, mock_ctx: AsyncMock, installed_clis: list[str]
    ) -> None:
        """elicit=False with all tasks having cli set → no elicit calls, even for YOLO."""
        tasks = [
            AgentTask(cli="gemini", prompt="do something useful here", execution_mode="yolo"),
        ]
        guard = ElicitationGuard(mock_ctx, installed_clis)
        result = await guard.check_batch(tasks, elicit=False)
        mock_ctx.elicit.assert_not_called()
        assert result[0].execution_mode == "yolo"
