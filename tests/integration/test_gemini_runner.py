# tests/integration/test_gemini_runner.py
"""Integration tests for GeminiRunner with real CLI binary.

Minimal set of tests that CANNOT be covered by unit/e2e tests (which mock
at the subprocess boundary). These validate:

1. Real CLI detection + capability resolution
2. Real API call → valid AgentResponse (happy path)
3. Real API error behavior (error path)

All other GeminiRunner behavior (command building, JSON parsing, retry logic,
progress reporting) is fully covered by unit tests in tests/unit/runners/test_gemini.py
and pipeline tests in tests/unit/test_server_gemini_pipeline.py.
"""

import re

import pytest

from nexus_mcp.cli_detector import detect_cli, get_cli_capabilities, get_cli_version
from nexus_mcp.runners.gemini import GeminiRunner
from nexus_mcp.types import AgentResponse
from tests.fixtures import PING_PROMPT, make_prompt_request

_SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+")


@pytest.mark.integration
class TestGeminiCLISmoke:
    """Validate real CLI detection, version parsing, and capability resolution."""

    def test_cli_detected_with_json_support(self, gemini_cli_available: str) -> None:
        """detect_cli finds binary, version parses as semver, supports_json is True."""
        cli_info = detect_cli("gemini")
        assert cli_info.found is True
        assert cli_info.path == gemini_cli_available

        version = get_cli_version("gemini")
        assert version is not None, "get_cli_version returned None"
        assert _SEMVER_PATTERN.match(version), f"Expected semver, got: {version!r}"

        capabilities = get_cli_capabilities("gemini", version)
        assert capabilities.found is True
        assert capabilities.supports_json is True, (
            f"Gemini CLI {version} does not support JSON — requires >= 0.6.0"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestGeminiHappyPath:
    """Real API call → valid AgentResponse."""

    async def test_run_returns_valid_response(self, gemini_runner: GeminiRunner) -> None:
        """run() with real API returns AgentResponse with output and metadata."""
        request = make_prompt_request(prompt=PING_PROMPT)
        response = await gemini_runner.run(request)

        assert isinstance(response, AgentResponse)
        assert response.cli == "gemini"
        assert len(response.output) > 0
        assert isinstance(response.metadata, dict)


@pytest.mark.integration
@pytest.mark.slow
class TestGeminiErrorPath:
    """Real API error behavior with invalid model."""

    async def test_invalid_model_raises_or_recovers(self, gemini_runner: GeminiRunner) -> None:
        """run() with nonexistent model raises SubprocessError or recovers gracefully."""
        from nexus_mcp.exceptions import SubprocessError

        request = make_prompt_request(prompt="ping", model="nonexistent-model-xyz-99")
        try:
            response = await gemini_runner.run(request)
            assert response.metadata.get("recovered_from_error") is True, (
                f"Expected SubprocessError or recovered_from_error=True, got: {response.output!r}"
            )
        except SubprocessError:
            pass  # Error propagation path — also correct
