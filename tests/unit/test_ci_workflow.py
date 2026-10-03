import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _job(name: str) -> str:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index(f"\n  {name}:\n")
    next_job = re.compile(r"\n  [a-z][a-z0-9_-]*:\n").search(workflow, start + 1)
    return workflow[start : next_job.start() if next_job else None]


def test_diff_coverage_uses_shared_action_with_nexus_policy() -> None:
    job = _job("coverage")

    assert re.search(
        r"uses: j7an/shared-workflows/actions/coverage@[0-9a-f]{40} # v[0-9]+\.[0-9]+\.[0-9]+",
        job,
    )
    assert "fetch-depth: 0" in job
    assert 'save-cache: "false"' in job
    assert "--cov=nexus_mcp" in job
    assert "--cov-report=xml" in job
    assert "report-path: coverage.xml" in job
    assert "base-sha: ${{ github.event.pull_request.base.sha }}" in job
    assert 'minimum: "90"' in job
    assert "source-paths: src/nexus_mcp/" in job
    assert "exclude-paths: src/nexus_mcp/__main__.py" in job
    assert "diff-cover-path: ${{ github.workspace }}/.venv/bin/diff-cover" in job


def test_test_matrix_does_not_measure_coverage() -> None:
    job = _job("test")

    assert "--cov" not in job
    assert "shared-workflows/actions/coverage" not in job
