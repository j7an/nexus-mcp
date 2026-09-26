# AGENTS.md - Guidelines for AI Coding Agents

## Project Overview

Nexus MCP is a Python 3.12+ stdio MCP server (FastMCP 4.0.x) exposing one tool, `prompt`, and
one resource, `nexus://backends`. Each `prompt` call runs one synchronous agent turn through a
backend: Claude via the Claude Agent SDK. Backends are optional extras. There is no nexus
persistence: conversations persist in each agent's own storage and are resumed by `session_id`.

## Build, Lint, Test Commands

```bash
# Setup (first time)
uv sync --all-extras --all-groups
uv run pre-commit install

# Run all tests (unit + e2e, fast)
uv run pytest

# Run single test file
uv run pytest tests/unit/test_server.py

# Run single test function
uv run pytest tests/unit/test_server.py::test_prompt_dispatches_validated_request

# Run with coverage (threshold: 90%)
uv run pytest --cov=nexus_mcp -v

# Integration tests (slow - real CLI binaries, run before PR only)
uv run pytest -m integration

# Type check
uv run mypy src/nexus_mcp

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Auto-fix lint issues
uv run ruff check --fix .

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Branch and Commit Guidance

- Never commit to `main` unless explicitly directed.
- For issue or feature work, create a branch or worktree from `origin/main`, keep all changes there, and prepare the result as a PR.
- Never commit spec or plan documents unless explicitly asked.
- Do not use `git add -A` or `git add .`; stage explicit paths.
- Ask before adding new dependencies or making destructive changes.

## Release Workflow Guidance

- Keep PyPI and TestPyPI Trusted Publishing jobs caller-owned in
  `.github/workflows/release.yml`. Do not replace them with
  `j7an/shared-workflows/.github/workflows/publish-pypi.yml`; PyPI does not
  authorize cross-repo reusable workflows as trusted-publisher workflows.
- When porting fixes from `shared-workflows`, use the caller-owned PyPI
  Trusted Publishing template and preserve Nexus-specific safeguards:
  `VERIFY_PYTHON: "3.13"`, `VERIFY_COMMAND: nexus-mcp --version`,
  `scripts/derive-published-version.sh` before upload, TestPyPI and PyPI
  environment URLs, and MCP Registry publishing gated on GitHub release success.
- Keep `setup-uv` caching disabled in the TestPyPI verification job while it
  intentionally skips checkout and creates the throwaway `.verify` project
  later. If caching is re-enabled, the job must first provide real dependency
  files for `setup-uv` to hash so release runs do not emit empty-workdir or
  cache-dependency warnings.
- Pin reusable `shared-workflows` callers to immutable commit SHAs with a
  version comment, but keep PyPI publishing inline unless PyPI explicitly
  supports cross-repo reusable workflows for Trusted Publishing.

## Python Code Style

### Modern Syntax (Python 3.12+)

- Use `str | None` instead of `Optional[str]`.
- Use `type` aliases instead of `TypeAlias`.
- Use `match` statements for complex conditionals.
- Do not use `from __future__ import annotations`.

### Formatting

- Line length: 100 characters.
- Double quotes for strings.
- Imports: standard library, third-party, local; sorted with blank lines between groups.
- Functions should be focused, shallow, and easy to scan.

### Naming Conventions

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private attributes: `_leading_underscore`
- Type aliases: `PascalCase`

### Docstrings

Public functions and classes use Google-style docstrings. Private methods can use concise
docstrings when they clarify intent.

## Imports

```python
from nexus_mcp.types import PromptResult
```

Public modules use `__all__` for re-exports.

## Types and Models

- Use Pydantic `BaseModel` for request/response types.
- Use `frozen=True` for immutable models.
- Define type aliases in `types.py`.
- Use `Protocol` for callback interfaces.

## Error Handling

Raise `fastmcp.exceptions.ToolError` with actionable text built from fixed strings and safe
enumerated fields (result subtype from a known set, HTTP status, exception class name). Never
copy provider free text into error messages.

## Testing

- `tests/unit/` — fast, isolated unit tests
- `tests/e2e/` — in-process MCP protocol tests via `Client(mcp)` in both protocol eras
- `tests/integration/` — slow, real CLI calls (`-m integration`)

Mock at the SDK boundary: patch `nexus_mcp.backends.claude.ClaudeSDKClient` (see
`tests/unit/backends/claude_fakes.py`). Server tests patch `backends.installed` / `backends.get`.

`asyncio_mode = "auto"` is configured, so async tests do not need
`@pytest.mark.asyncio`.

Run targeted tests after code changes when practical.

## Module Responsibilities

- `server.py` — FastMCP instance, `prompt` tool, `nexus://backends`, input validation, timeout
- `types.py` — `PromptRequest`, `PromptResult`, `BackendInfo`, `Profile`, `BackendName`
- `backends/__init__.py` — registry: name → module; installed = SDK extra importable
- `backends/claude.py` — Claude Agent SDK turn (`run`, `info`)
- `backends/claude_policy.py` — permission profiles (`decide`) and SDK options

A backend module defines `async def run(req: PromptRequest, on_session=...) -> PromptResult`
(calling `on_session(session_id)` once the provider reports it) and
`async def info() -> BackendInfo`, plus an entry in `backends._SDK_PACKAGES`.

## Search

- Prefer `rg` for text search and `rg --files` for file listing.
- Use `ast-grep --lang [language] -p '<pattern>'` for syntax-aware code search.

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`. Run manually:

```bash
uv run pre-commit run --all-files
```
