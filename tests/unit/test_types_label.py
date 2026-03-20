# tests/unit/test_types_label.py
"""Test that AgentTask.to_request() forwards label into context."""

from tests.fixtures import make_agent_task


class TestToRequestLabelForwarding:
    def test_label_injected_into_context(self):
        task = make_agent_task(cli="opencode_server", label="my-task")
        request = task.to_request()
        assert request.context.get("_nexus_label") == "my-task"

    def test_no_label_no_injection(self):
        task = make_agent_task(cli="opencode_server", label=None)
        request = task.to_request()
        assert "_nexus_label" not in request.context

    def test_existing_context_preserved(self):
        task = make_agent_task(
            cli="opencode_server",
            label="my-task",
            context={"project": "nexus"},
        )
        request = task.to_request()
        assert request.context["project"] == "nexus"
        assert request.context["_nexus_label"] == "my-task"
