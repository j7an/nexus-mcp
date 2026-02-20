import pytest
from pydantic import ValidationError

from nexus_mcp.types import (
    AgentResponse,
    AgentTask,
    AgentTaskResult,
    MultiPromptResponse,
    PromptRequest,
    SubprocessResult,
)


def test_prompt_request_valid():
    req = PromptRequest(agent="gemini", prompt="Hello")
    assert req.agent == "gemini"
    assert req.prompt == "Hello"
    assert req.context == {}


def test_prompt_request_empty_prompt_fails():
    with pytest.raises(ValidationError):
        PromptRequest(agent="gemini", prompt="")


def test_prompt_request_default_execution_mode():
    req = PromptRequest(agent="gemini", prompt="Hello")
    assert req.execution_mode == "default"


def test_prompt_request_yolo_mode():
    req = PromptRequest(agent="gemini", prompt="Hello", execution_mode="yolo")
    assert req.execution_mode == "yolo"


def test_prompt_request_invalid_execution_mode():
    with pytest.raises(ValidationError):
        PromptRequest(agent="gemini", prompt="Hello", execution_mode="invalid")


def test_agent_response_structure():
    resp = AgentResponse(
        agent="gemini", output="test output", raw_output='{"response": "test output"}'
    )
    assert resp.agent == "gemini"
    assert resp.output == "test output"
    assert resp.metadata == {}


def test_subprocess_result_structure():
    result = SubprocessResult(stdout="output", stderr="", returncode=0)
    assert result.stdout == "output"
    assert result.stderr == ""
    assert result.returncode == 0


def test_subprocess_result_nonzero_returncode():
    result = SubprocessResult(stdout="", stderr="error", returncode=1)
    assert result.returncode == 1
    assert result.stderr == "error"


def test_prompt_request_with_model():
    """Model field should accept valid model names."""
    req = PromptRequest(agent="gemini", prompt="Hello", model="gemini-2.5-flash")
    assert req.model == "gemini-2.5-flash"


def test_prompt_request_default_model_is_none():
    """Model field should default to None (uses CLI default)."""
    req = PromptRequest(agent="gemini", prompt="Hello")
    assert req.model is None


def test_prompt_request_empty_string_model_fails():
    """Empty string should fail validation (prevent malformed commands)."""
    with pytest.raises(ValidationError):
        PromptRequest(agent="gemini", prompt="Hello", model="")


# Phase 3.7: File References tests
def test_prompt_request_with_file_refs():
    """PromptRequest accepts optional file_refs list."""
    request = PromptRequest(
        agent="gemini", prompt="Analyze this code", file_refs=["src/main.py", "tests/test_main.py"]
    )
    assert request.file_refs == ["src/main.py", "tests/test_main.py"]


def test_prompt_request_file_refs_default_empty():
    """PromptRequest.file_refs defaults to empty list."""
    request = PromptRequest(agent="gemini", prompt="test")
    assert request.file_refs == []


def test_prompt_request_file_refs_must_be_strings():
    """PromptRequest.file_refs must contain only strings."""
    with pytest.raises(ValidationError):
        PromptRequest(
            agent="gemini",
            prompt="test",
            file_refs=["valid.py", 123, None],  # Invalid types
        )


def test_agent_response_with_metadata_adds_keys():
    """with_metadata() returns a new response with additional metadata keys."""
    original = AgentResponse(agent="gemini", output="hello", raw_output="{}", metadata={"k": 1})
    updated = original.with_metadata(new_key="value", count=42)

    assert updated.metadata == {"k": 1, "new_key": "value", "count": 42}


def test_agent_response_with_metadata_preserves_other_fields():
    """with_metadata() preserves agent, output, and raw_output unchanged."""
    original = AgentResponse(agent="gemini", output="hello", raw_output="{}")
    updated = original.with_metadata(x=1)

    assert updated.agent == original.agent
    assert updated.output == original.output
    assert updated.raw_output == original.raw_output


def test_agent_response_with_metadata_does_not_mutate_original():
    """with_metadata() returns a new object; the original is unchanged."""
    original = AgentResponse(agent="gemini", output="hello", raw_output="{}", metadata={"k": 1})
    original.with_metadata(k=999)

    assert original.metadata == {"k": 1}


# ---------------------------------------------------------------------------
# AgentTask / AgentTaskResult / MultiPromptResponse tests
# ---------------------------------------------------------------------------


def test_agent_task_requires_agent_and_prompt():
    task = AgentTask(agent="gemini", prompt="Hello")
    assert task.agent == "gemini"
    assert task.prompt == "Hello"
    assert task.label is None
    assert task.execution_mode == "default"


def test_agent_task_rejects_empty_agent():
    with pytest.raises(ValidationError):
        AgentTask(agent="", prompt="Hello")


def test_agent_task_rejects_empty_prompt():
    with pytest.raises(ValidationError):
        AgentTask(agent="gemini", prompt="")


def test_agent_task_result_success():
    result = AgentTaskResult(label="gemini", output="response text")
    assert result.success is True
    assert result.output == "response text"
    assert result.error is None


def test_agent_task_result_error():
    result = AgentTaskResult(label="gemini", error="something failed")
    assert result.success is False
    assert result.error == "something failed"
    assert result.output is None


def test_agent_task_result_rejects_both_output_and_error():
    with pytest.raises(ValidationError):
        AgentTaskResult(label="gemini", output="ok", error="fail")


def test_agent_task_result_rejects_neither_output_nor_error():
    with pytest.raises(ValidationError):
        AgentTaskResult(label="gemini")


def test_multi_prompt_response_counts():
    results = [
        AgentTaskResult(label="a", output="ok"),
        AgentTaskResult(label="b", error="fail"),
        AgentTaskResult(label="c", output="ok"),
    ]
    response = MultiPromptResponse(results=results)
    assert response.total == 3
    assert response.succeeded == 2
    assert response.failed == 1
