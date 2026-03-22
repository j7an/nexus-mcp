# tests/unit/test_server_gemini_pipeline.py
"""Pipeline integration tests: server tools → RunnerFactory → GeminiRunner → subprocess.

Mocks only at asyncio.create_subprocess_exec — all layers above run for real:
    prompt()/batch_prompt() → RunnerFactory → GeminiRunner → build_command
        → run_subprocess → [MOCK] → parse_output

This validates the full request/response path that server tests (mock at RunnerFactory)
and runner tests (mock at subprocess) independently miss.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

from nexus_mcp.server import batch_prompt, prompt
from tests.fixtures import GEMINI_NOISY_STDOUT, create_mock_process, make_agent_task

# ---------------------------------------------------------------------------
# JSON response builders
# ---------------------------------------------------------------------------


def _gemini_json(output: str, stats: dict | None = None) -> str:
    """Build a Gemini CLI JSON response string."""
    data: dict = {"response": output}
    if stats:
        data["stats"] = stats
    return json.dumps(data)


def _gemini_error_json(code: int, message: str, status: str) -> str:
    """Build a Gemini API error JSON string."""
    return json.dumps({"error": {"code": code, "message": message, "status": status}})


# ---------------------------------------------------------------------------
# Class 1: prompt() → GeminiRunner pipeline (10 tests)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_cli_detection")
class TestPromptGeminiPipeline:
    """Tests for the full prompt() → GeminiRunner pipeline.

    Mocks only asyncio.create_subprocess_exec. Everything above runs for real:
        prompt() → batch_prompt() → RunnerFactory.create("gemini") → GeminiRunner
            → build_command() → run_subprocess() → [MOCK] → parse_output()
    """

    async def test_success_returns_parsed_output(self, mock_subprocess):
        """Full success path: subprocess returns valid JSON → prompt() returns output text."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("Hello from Gemini"))

        result = await prompt("gemini", "Say hello")

        assert result == "Hello from Gemini"
        assert mock_subprocess.call_count == 1
        args = list(mock_subprocess.call_args.args)
        assert "-p" in args
        assert "--output-format" in args
        assert "json" in args

    async def test_model_reaches_subprocess_args(self, mock_subprocess):
        """model parameter is forwarded as --model flag in the subprocess command."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))

        await prompt("gemini", "test", model="gemini-2.5-flash")

        args = list(mock_subprocess.call_args.args)
        assert "--model" in args
        assert "gemini-2.5-flash" in args

    async def test_yolo_mode_reaches_subprocess_args(self, mock_subprocess):
        """execution_mode='yolo' adds --yolo to subprocess args."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))

        await prompt("gemini", "test", execution_mode="yolo")

        args = list(mock_subprocess.call_args.args)
        assert "--yolo" in args

    async def test_parse_error_raises_tool_error(self, mock_subprocess):
        """Invalid JSON stdout → ParseError → ToolError with [ParseError] prefix."""
        mock_subprocess.return_value = create_mock_process(stdout="not valid json", returncode=0)

        with pytest.raises(ToolError, match=r"\[ParseError\]"):
            await prompt("gemini", "test")

    async def test_401_raises_tool_error(self, mock_subprocess):
        """Non-retryable API error (401) → SubprocessError → ToolError with [SubprocessError]."""
        error_json = _gemini_error_json(
            401,
            "UNAUTHENTICATED: Request had invalid authentication credentials.",
            "UNAUTHENTICATED",
        )
        mock_subprocess.return_value = create_mock_process(stdout=error_json, returncode=1)

        with pytest.raises(ToolError, match=r"\[SubprocessError\]"):
            await prompt("gemini", "test")

    async def test_429_retries_then_succeeds(self, mock_subprocess, fast_retry_sleep):
        """HTTP 429 rate limit triggers retry; second attempt succeeds."""
        error_json = _gemini_error_json(429, "Resource exhausted", "RESOURCE_EXHAUSTED")
        mock_subprocess.side_effect = [
            create_mock_process(stdout=error_json, returncode=1),
            create_mock_process(stdout=_gemini_json("ok after retry")),
        ]

        result = await prompt("gemini", "test", max_retries=2)

        assert result == "ok after retry"
        assert mock_subprocess.call_count == 2

    async def test_noisy_stdout_parses_correctly(self, mock_subprocess):
        """Node.js warnings before JSON are stripped; response field is extracted correctly."""
        mock_subprocess.return_value = create_mock_process(stdout=GEMINI_NOISY_STDOUT, returncode=0)

        result = await prompt("gemini", "test")

        assert result == "test output"

    async def test_recovery_from_nonzero_exit_with_valid_json(self, mock_subprocess):
        """Non-zero exit with valid response JSON → recovery path → output returned."""
        mock_subprocess.return_value = create_mock_process(
            stdout=_gemini_json("recovered output"), returncode=1
        )

        result = await prompt("gemini", "test")

        assert result == "recovered output"

    async def test_503_exhausts_retries_raises_tool_error(self, mock_subprocess, fast_retry_sleep):
        """HTTP 503 that always fails → exhausts all retries → ToolError with [RetryableError]."""
        error_json = _gemini_error_json(503, "Service unavailable", "UNAVAILABLE")
        mock_subprocess.return_value = create_mock_process(stdout=error_json, returncode=1)

        with pytest.raises(ToolError, match=r"\[RetryableError\]"):
            await prompt("gemini", "test", max_retries=3)

        assert mock_subprocess.call_count == 3


# ---------------------------------------------------------------------------
# Class 2: batch_prompt() → GeminiRunner pipeline (7 tests)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("mock_cli_detection")
class TestBatchPromptGeminiPipeline:
    """Tests for the full batch_prompt() → GeminiRunner pipeline.

    Same mock boundary as TestPromptGeminiPipeline, but exercises:
    - Multi-task parallelism via asyncio.gather
    - Per-task label assignment
    - Per-task model and execution mode
    - Partial failure handling

    Note: batch_prompt() uses asyncio.gather() — tasks run concurrently. Tests that need
    per-task subprocess behavior use side_effect functions keyed on the prompt argument
    rather than list-based side_effect, since call ordering within gather is non-deterministic.
    """

    async def test_all_succeed_with_correct_labels(self, mock_subprocess):
        """Two tasks both succeed; auto-assigned labels are 'gemini' and 'gemini-2'."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))
        tasks = [make_agent_task(prompt="first"), make_agent_task(prompt="second")]

        response = await batch_prompt(tasks=tasks)

        assert response.succeeded == 2
        labels = {r.label for r in response.results}
        assert "gemini" in labels
        assert "gemini-2" in labels

    async def test_mixed_success_and_parse_error(self, mock_subprocess):
        """One task returns valid JSON; one returns invalid JSON → succeeded=1, failed=1."""

        def side_effect(*args, **kwargs):
            cmd = list(args)
            prompt_text = cmd[cmd.index("-p") + 1]
            if "succeed" in prompt_text:
                return create_mock_process(stdout=_gemini_json("ok"))
            return create_mock_process(stdout="not valid json", returncode=0)

        mock_subprocess.side_effect = side_effect
        tasks = [make_agent_task(prompt="please succeed"), make_agent_task(prompt="will fail")]

        response = await batch_prompt(tasks=tasks)

        assert response.succeeded == 1
        assert response.failed == 1
        failed_result = next(r for r in response.results if r.error)
        assert failed_result.error_type == "ParseError"

    async def test_different_models_per_task(self, mock_subprocess):
        """Each task's model flag reaches the subprocess independently."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))
        tasks = [
            make_agent_task(prompt="task1", model="gemini-2.5-flash"),
            make_agent_task(prompt="task2", model="gemini-2.0-pro"),
        ]

        response = await batch_prompt(tasks=tasks)

        assert response.succeeded == 2
        all_args = [arg for call in mock_subprocess.call_args_list for arg in call.args]
        assert "gemini-2.5-flash" in all_args
        assert "gemini-2.0-pro" in all_args

    async def test_different_execution_modes_per_task(self, mock_subprocess):
        """--yolo appears in subprocess args for yolo task; default task has no extra flags."""
        mock_subprocess.return_value = create_mock_process(stdout=_gemini_json("ok"))
        tasks = [
            make_agent_task(prompt="default task"),
            make_agent_task(prompt="yolo task", execution_mode="yolo"),
        ]

        response = await batch_prompt(tasks=tasks)

        assert response.succeeded == 2
        all_args = [arg for call in mock_subprocess.call_args_list for arg in call.args]
        assert "--yolo" in all_args

    async def test_retryable_error_retries_and_succeeds(self, mock_subprocess, fast_retry_sleep):
        """One task fails with 429 once then succeeds; total subprocess calls = 3."""
        call_counts: dict[str, int] = {}

        def side_effect(*args, **kwargs):
            cmd = list(args)
            prompt_text = cmd[cmd.index("-p") + 1]
            count = call_counts.get(prompt_text, 0)
            call_counts[prompt_text] = count + 1
            if prompt_text == "retry me" and count == 0:
                return create_mock_process(
                    stdout=_gemini_error_json(429, "Rate limited", "RESOURCE_EXHAUSTED"),
                    returncode=1,
                )
            return create_mock_process(stdout=_gemini_json("ok"))

        mock_subprocess.side_effect = side_effect
        tasks = [
            make_agent_task(prompt="retry me", max_retries=2),
            make_agent_task(prompt="no retry needed"),
        ]

        response = await batch_prompt(tasks=tasks)

        assert response.succeeded == 2
        assert mock_subprocess.call_count == 3  # 2 calls for "retry me" + 1 for "no retry needed"

    async def test_max_retries_override_per_task(self, mock_subprocess, fast_retry_sleep):
        """Per-task max_retries controls attempt count: task_a=1, task_b=2 → 3 total calls."""
        error_json = _gemini_error_json(429, "Rate limited", "RESOURCE_EXHAUSTED")
        mock_subprocess.return_value = create_mock_process(stdout=error_json, returncode=1)
        tasks = [
            make_agent_task(prompt="task a", max_retries=1),
            make_agent_task(prompt="task b", max_retries=2),
        ]

        response = await batch_prompt(tasks=tasks)

        assert response.failed == 2
        assert mock_subprocess.call_count == 3  # 1 attempt (task a) + 2 attempts (task b)

    async def test_ctx_info_reports_partial_success(self, mock_subprocess, ctx):
        """ctx.info() called twice; completion message includes actual success/fail counts."""

        def side_effect(*args, **kwargs):
            cmd = list(args)
            prompt_text = cmd[cmd.index("-p") + 1]
            if "succeed" in prompt_text:
                return create_mock_process(stdout=_gemini_json("ok"))
            return create_mock_process(stdout="bad json", returncode=0)

        mock_subprocess.side_effect = side_effect
        tasks = [make_agent_task(prompt="please succeed"), make_agent_task(prompt="will fail")]

        await batch_prompt(tasks=tasks, ctx=ctx)

        # ctx.info called for: batch start, "Running gemini" per task, batch complete
        assert ctx.info.await_count >= 2
        completion_msg = ctx.info.call_args_list[-1].args[0]
        assert "1/2" in completion_msg
