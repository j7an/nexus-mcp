"""Pipeline integration tests for runner enhancements.

Verifies all runner features work together in a single execution:
1. Output limiting (truncation to configured byte limit)
2. Error recovery (parse stdout despite non-zero exit)
3. File references (appended to prompt)
4. Environment variable configuration (custom path, model, limits)
"""

import os
from unittest.mock import patch

from nexus_mcp.runners.gemini import GeminiRunner
from tests.fixtures import create_mock_process, make_prompt_request


@patch.dict(
    os.environ,
    {
        "NEXUS_GEMINI_PATH": "/custom/gemini",
        "NEXUS_GEMINI_MODEL": "gemini-2.5-flash",
        "NEXUS_OUTPUT_LIMIT_BYTES": "1000",
    },
)
@patch("nexus_mcp.process.asyncio.create_subprocess_exec")
async def test_all_enhancements_together(mock_exec):
    """Verify all 4 enhancements work together without conflicts."""
    # Setup env vars (Step 4)

    # Mock large output that will be truncated
    large_output = "x" * 5000
    mock_exec.return_value = create_mock_process(
        stdout='{"response": "' + large_output + '"}',
        returncode=1,  # Error recovery scenario (Step 2)
    )

    runner = GeminiRunner()
    request = make_prompt_request(
        prompt="Analyze code",
        file_refs=["src/main.py"],  # File refs (Step 3)
    )

    response = await runner.run(request)

    # Verify all features active:
    # 1. Custom CLI path used (env config)
    args = mock_exec.call_args[0]
    assert args[0] == "/custom/gemini"

    # 2. File refs appended to prompt
    assert "src/main.py" in args[2]

    # 3. Error recovery succeeded
    assert response.metadata.get("recovered_from_error") is True

    # 4. Output was truncated
    assert response.metadata.get("truncated") is True
    assert len(response.output.encode("utf-8")) <= 1000
