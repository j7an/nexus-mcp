import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _coverage_step() -> str:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("      - name: Check diff coverage")
    end = workflow.index("\n\n  lint:", start)
    return workflow[start:end]


def test_diff_coverage_uses_shared_action_with_nexus_policy() -> None:
    step = _coverage_step()

    assert re.search(
        r"uses: j7an/shared-workflows/actions/coverage@[0-9a-f]{40} # v[0-9]+\.[0-9]+\.[0-9]+",
        step,
    )
    assert re.search(
        r"if: github\.event_name == 'pull_request' && matrix\.os == 'ubuntu-latest' "
        r"&& matrix\.python-version == '3\.13'\n",
        step,
    )
    assert "report-path: coverage.xml" in step
    assert "base-sha: ${{ github.event.pull_request.base.sha }}" in step
    assert 'minimum: "90"' in step
    assert "source-paths: src/nexus_mcp/" in step
    assert "exclude-paths: src/nexus_mcp/__main__.py" in step
    assert "diff-cover-path: ${{ github.workspace }}/.venv/bin/diff-cover" in step
