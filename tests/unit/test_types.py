import pytest
from pydantic import ValidationError


def test_prompt_request_valid():
    from nexus_mcp.types import PromptRequest

    req = PromptRequest(agent="gemini", prompt="Hello")
    assert req.agent == "gemini"
    assert req.prompt == "Hello"
    assert req.context == {}


def test_prompt_request_empty_prompt_fails():
    from nexus_mcp.types import PromptRequest

    with pytest.raises(ValidationError):
        PromptRequest(agent="gemini", prompt="")


def test_prompt_request_default_execution_mode():
    from nexus_mcp.types import PromptRequest

    req = PromptRequest(agent="gemini", prompt="Hello")
    assert req.execution_mode == "default"


def test_prompt_request_yolo_mode():
    from nexus_mcp.types import PromptRequest

    req = PromptRequest(agent="gemini", prompt="Hello", execution_mode="yolo")
    assert req.execution_mode == "yolo"


def test_prompt_request_invalid_execution_mode():
    from nexus_mcp.types import PromptRequest

    with pytest.raises(ValidationError):
        PromptRequest(agent="gemini", prompt="Hello", execution_mode="invalid")


def test_agent_response_structure():
    from nexus_mcp.types import AgentResponse

    resp = AgentResponse(
        agent="gemini", output="test output", raw_output='{"response": "test output"}'
    )
    assert resp.agent == "gemini"
    assert resp.output == "test output"
    assert resp.metadata == {}


def test_subprocess_result_structure():
    from nexus_mcp.types import SubprocessResult

    result = SubprocessResult(stdout="output", stderr="", returncode=0)
    assert result.stdout == "output"
    assert result.stderr == ""
    assert result.returncode == 0


def test_subprocess_result_nonzero_returncode():
    from nexus_mcp.types import SubprocessResult

    result = SubprocessResult(stdout="", stderr="error", returncode=1)
    assert result.returncode == 1
    assert result.stderr == "error"


def test_prompt_request_with_model():
    """Model field should accept valid model names."""
    from nexus_mcp.types import PromptRequest

    req = PromptRequest(agent="gemini", prompt="Hello", model="gemini-2.5-flash")
    assert req.model == "gemini-2.5-flash"


def test_prompt_request_default_model_is_none():
    """Model field should default to None (uses CLI default)."""
    from nexus_mcp.types import PromptRequest

    req = PromptRequest(agent="gemini", prompt="Hello")
    assert req.model is None


def test_prompt_request_empty_string_model_fails():
    """Empty string should fail validation (prevent malformed commands)."""
    from nexus_mcp.types import PromptRequest

    with pytest.raises(ValidationError):
        PromptRequest(agent="gemini", prompt="Hello", model="")


def test_prompt_request_with_session_id():
    """Session ID field should accept valid session IDs."""
    from nexus_mcp.types import PromptRequest

    req = PromptRequest(agent="claude", prompt="Continue work", session_id="session-123")
    assert req.session_id == "session-123"


def test_prompt_request_default_session_id_is_none():
    """Session ID field should default to None (new session)."""
    from nexus_mcp.types import PromptRequest

    req = PromptRequest(agent="claude", prompt="Hello")
    assert req.session_id is None


def test_prompt_request_empty_string_session_id_fails():
    """Empty string should fail validation (prevent malformed commands)."""
    from nexus_mcp.types import PromptRequest

    with pytest.raises(ValidationError):
        PromptRequest(agent="claude", prompt="Hello", session_id="")
