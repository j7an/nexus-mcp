from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


def _workflow_text() -> str:
    return RELEASE_WORKFLOW.read_text(encoding="utf-8")


def test_testpypi_verifier_disables_setup_uv_cache() -> None:
    workflow = _workflow_text()

    job_start = workflow.index("  verify-testpypi:")
    next_job = workflow.index("\n  publish-pypi:", job_start)
    verify_job = workflow[job_start:next_job]

    setup_start = verify_job.index("      - name: Set up uv")
    next_step = verify_job.index("\n      - name:", setup_start + 1)
    setup_step = verify_job[setup_start:next_step]

    assert "uses: astral-sh/setup-uv@" in setup_step
    assert "with:" in setup_step
    assert "enable-cache: false" in setup_step
