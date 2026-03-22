# Release Guide

Quick reference for maintainers releasing a new version of `nexus-mcp`.

## Prerequisites

These are already configured. Verify once if something seems broken:

- [ ] CI is passing on `main`
- [ ] PyPI Trusted Publisher configured for the `pypi` environment
  (Settings → Environments → `pypi` → required reviewer set)
- [ ] TestPyPI Trusted Publisher configured for the `testpypi` environment
- [ ] Both environments use OIDC — no API tokens needed

For initial setup details, see `docs/phase-6-release-workflow.md`.

---

## Releasing a New Version

Run these commands in order — the inline comments explain each step:

    # 1. Bump version and create release branch
    git checkout main && git pull origin main
    uv version --bump patch   # or minor / major
    VERSION=$(uv version --short)
    git checkout -b "release/v${VERSION}"

    # 2. Commit and push the release branch
    git add pyproject.toml uv.lock
    git commit -m "chore(release): v${VERSION}"
    git push -u origin "release/v${VERSION}"

    # 3. Create PR, wait for CI, merge
    gh pr create --title "chore(release): v${VERSION}" --body "Version bump to ${VERSION}"
    # After CI passes, merge via GitHub UI or:
    gh pr merge --squash --delete-branch "release/v${VERSION}"

    # 4. Tag the merge commit on main — workflow triggers automatically
    git checkout main && git pull origin main
    git tag "v${VERSION}"
    git push origin "v${VERSION}"

---

## What Happens Next (Automated)

The `release.yml` workflow runs five jobs in sequence:

| Job | What it does |
|-----|--------------|
| `build` | Verifies the tag is on `main`, builds sdist + wheel via `uv build`, uploads artifacts |
| `publish-testpypi` | Publishes to TestPyPI, waits 30 s, installs and smoke-tests the package |
| `publish-pypi` | **Waits for a required reviewer to approve** the `pypi` environment, then publishes |
| `github-release` | Creates a **draft** GitHub Release with auto-generated notes and dist assets attached |
| `publish-mcp-registry` | Patches `server.json` version from tag, authenticates via OIDC, publishes metadata to MCP Registry |

Monitor progress at:
`https://github.com/j7an/nexus-mcp/actions`

---

## After the Workflow Completes

- [ ] Approve the `pypi` environment deployment when GitHub prompts you
- [ ] Verify the live package: `pip install "nexus-mcp==${VERSION}"`
- [ ] Smoke-test: `python -c "import nexus_mcp; print(nexus_mcp.__version__)"`
- [ ] Open the draft GitHub Release, review auto-generated notes, and click **Publish release**
- [ ] Verify the MCP Registry listing

---

## Pre-release Versions

Tags containing a hyphen are automatically marked as pre-releases by the workflow:

    git tag v1.0.0-rc1
    git push origin v1.0.0-rc1

Examples: `v1.0.0-alpha1`, `v1.0.0-beta2`, `v1.0.0-rc1`

---

## Recovering from a Failed Release

### Build job failed

Fix the issue on `main`, then re-push the tag:

    git tag -d "v${VERSION}"
    git push origin ":refs/tags/v${VERSION}"
    # fix the issue, merge to main
    git tag "v${VERSION}" && git push origin "v${VERSION}"

### Published to TestPyPI but PyPI failed

The wheel and sdist are already uploaded to TestPyPI (immutable). You can publish
to PyPI manually using the artifacts from the failed workflow run, or bump to a
patch version (`X.Y.Z+1`) and re-release.

### GitHub Release not created

The `github-release` job only runs if `publish-pypi` succeeds. If it was skipped,
create the release manually:

    gh release create "v${VERSION}" dist/* \
      --title "v${VERSION}" \
      --generate-notes \
      --draft
