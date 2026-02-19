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


# Phase 3.7: File References tests
def test_prompt_request_with_file_refs():
    """PromptRequest accepts optional file_refs list."""
    from nexus_mcp.types import PromptRequest

    request = PromptRequest(
        agent="gemini", prompt="Analyze this code", file_refs=["src/main.py", "tests/test_main.py"]
    )
    assert request.file_refs == ["src/main.py", "tests/test_main.py"]


def test_prompt_request_file_refs_default_empty():
    """PromptRequest.file_refs defaults to empty list."""
    from nexus_mcp.types import PromptRequest

    request = PromptRequest(agent="gemini", prompt="test")
    assert request.file_refs == []


def test_prompt_request_file_refs_must_be_strings():
    """PromptRequest.file_refs must contain only strings."""
    from nexus_mcp.types import PromptRequest

    with pytest.raises(ValidationError):
        PromptRequest(
            agent="gemini",
            prompt="test",
            file_refs=["valid.py", 123, None],  # Invalid types
        )


def test_agent_response_with_metadata_adds_keys():
    """with_metadata() returns a new response with additional metadata keys."""
    from nexus_mcp.types import AgentResponse

    original = AgentResponse(agent="gemini", output="hello", raw_output="{}", metadata={"k": 1})
    updated = original.with_metadata(new_key="value", count=42)

    assert updated.metadata == {"k": 1, "new_key": "value", "count": 42}


def test_agent_response_with_metadata_preserves_other_fields():
    """with_metadata() preserves agent, output, and raw_output unchanged."""
    from nexus_mcp.types import AgentResponse

    original = AgentResponse(agent="gemini", output="hello", raw_output="{}")
    updated = original.with_metadata(x=1)

    assert updated.agent == original.agent
    assert updated.output == original.output
    assert updated.raw_output == original.raw_output


def test_agent_response_with_metadata_does_not_mutate_original():
    """with_metadata() returns a new object; the original is unchanged."""
    from nexus_mcp.types import AgentResponse

    original = AgentResponse(agent="gemini", output="hello", raw_output="{}", metadata={"k": 1})
    original.with_metadata(k=999)

    assert original.metadata == {"k": 1}


# ---------------------------------------------------------------------------
# AgentTask / AgentTaskResult / MultiPromptResponse tests
# ---------------------------------------------------------------------------


def test_agent_task_requires_agent_and_prompt():
    from nexus_mcp.types import AgentTask

    task = AgentTask(agent="gemini", prompt="Hello")
    assert task.agent == "gemini"
    assert task.prompt == "Hello"
    assert task.label is None
    assert task.execution_mode == "default"


def test_agent_task_rejects_empty_agent():
    from nexus_mcp.types import AgentTask

    with pytest.raises(ValidationError):
        AgentTask(agent="", prompt="Hello")


def test_agent_task_rejects_empty_prompt():
    from nexus_mcp.types import AgentTask

    with pytest.raises(ValidationError):
        AgentTask(agent="gemini", prompt="")


def test_agent_task_result_success():
    from nexus_mcp.types import AgentTaskResult

    result = AgentTaskResult(label="gemini", output="response text")
    assert result.success is True
    assert result.output == "response text"
    assert result.error is None


def test_agent_task_result_error():
    from nexus_mcp.types import AgentTaskResult

    result = AgentTaskResult(label="gemini", error="something failed")
    assert result.success is False
    assert result.error == "something failed"
    assert result.output is None


def test_agent_task_result_rejects_both_output_and_error():
    from nexus_mcp.types import AgentTaskResult

    with pytest.raises(ValidationError):
        AgentTaskResult(label="gemini", output="ok", error="fail")


def test_agent_task_result_rejects_neither_output_nor_error():
    from nexus_mcp.types import AgentTaskResult

    with pytest.raises(ValidationError):
        AgentTaskResult(label="gemini")


def test_multi_prompt_response_counts():
    from nexus_mcp.types import AgentTaskResult, MultiPromptResponse

    results = [
        AgentTaskResult(label="a", output="ok"),
        AgentTaskResult(label="b", error="fail"),
        AgentTaskResult(label="c", output="ok"),
    ]
    response = MultiPromptResponse(results=results)
    assert response.total == 3
    assert response.succeeded == 2
    assert response.failed == 1
