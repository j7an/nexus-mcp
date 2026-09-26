import json
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


def test_mcp_publisher_download_is_pinned_and_checksum_verified() -> None:
    workflow = _workflow_text()

    step_start = workflow.index("      - name: Install mcp-publisher")
    next_step = workflow.index("\n      - name:", step_start + 1)
    install_step = workflow[step_start:next_step]

    assert "releases/latest" not in install_step
    assert "releases/download/" in install_step
    assert "sha256sum --check" in install_step


def test_readme_has_registry_ownership_marker() -> None:
    server_name = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))["name"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"<!-- mcp-name: {server_name} -->" in readme
