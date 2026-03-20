# tests/unit/runners/test_opencode_server.py
"""Tests for OpenCodeServerRunner.

Mock boundary: httpx.AsyncClient — all HTTP calls mocked.
"""

import pytest

from nexus_mcp.runners.opencode_server import OpenCodeServerRunner
from tests.fixtures import make_prompt_request


def make_server_runner() -> OpenCodeServerRunner:
    """Create an OpenCodeServerRunner using autouse cli_detection_mocks."""
    return OpenCodeServerRunner()


class TestOpenCodeServerRunnerInit:
    """Test runner construction and ABC stubs."""

    def test_agent_name(self):
        runner = make_server_runner()
        assert runner.AGENT_NAME == "opencode_server"

    def test_build_command_raises(self):
        runner = make_server_runner()
        request = make_prompt_request(cli="opencode_server")
        with pytest.raises(NotImplementedError):
            runner.build_command(request)

    def test_parse_output_raises(self):
        runner = make_server_runner()
        with pytest.raises(NotImplementedError):
            runner.parse_output("stdout", "stderr")
