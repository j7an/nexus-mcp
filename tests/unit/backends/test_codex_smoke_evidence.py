"""Focused guards for native Codex rollout evidence used by smoke coverage."""

import pytest

from tests.integration.test_codex_smoke import (
    _correlated_execution_outputs,
    _is_permission_denial_with_nonzero_exit,
)


def _call(kind: str, name: str, call_id: str, command: str) -> dict[str, object]:
    field = "input" if kind == "custom_tool_call" else "arguments"
    return {
        "type": "response_item",
        "payload": {"type": kind, "name": name, "call_id": call_id, field: command},
    }


def _output(kind: str, call_id: str, output: object) -> dict[str, object]:
    return {
        "type": "response_item",
        "payload": {"type": kind, "call_id": call_id, "output": output},
    }


@pytest.mark.parametrize(
    ("call_kind", "output_kind", "command"),
    [
        ("function_call", "function_call_output", '{"cmd":"touch narrowed.txt"}'),
        (
            "custom_tool_call",
            "custom_tool_call_output",
            'tools.exec_command({cmd:"touch narrowed.txt"})',
        ),
    ],
)
def test_correlates_execution_call_with_its_native_output(call_kind, output_kind, command):
    records = [
        _call(
            call_kind, "exec_command" if call_kind == "function_call" else "exec", "target", command
        ),
        _output(output_kind, "target", {"exit_code": 1, "output": "Operation not permitted"}),
    ]
    outputs = _correlated_execution_outputs(records, "touch narrowed.txt")
    assert len(outputs) == 1
    assert _is_permission_denial_with_nonzero_exit(outputs[0])


@pytest.mark.parametrize(
    "records",
    [
        [
            {
                "type": "response_item",
                "payload": {"type": "message", "content": "touch narrowed.txt"},
            }
        ],
        [_call("function_call", "read_file", "target", '{"path":"narrowed.txt"}')],
        [
            _call("function_call", "exec_command", "target", '{"cmd":"touch narrowed.txt"}'),
            _output(
                "function_call_output",
                "other",
                {"exit_code": 1, "output": "Operation not permitted"},
            ),
        ],
    ],
)
def test_rejects_prose_non_execution_and_mismatched_output(records):
    assert _correlated_execution_outputs(records, "touch narrowed.txt") == []


@pytest.mark.parametrize(
    "output",
    [
        {"exit_code": 0, "output": "Operation not permitted"},
        {"output": "Operation not permitted"},
        {"exit_code": 1, "output": "completed"},
    ],
)
def test_requires_nonzero_exit_and_permission_denial(output):
    assert not _is_permission_denial_with_nonzero_exit(str(output))
