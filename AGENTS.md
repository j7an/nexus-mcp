# AGENTS.md - Guidelines for AI Coding Agents

## Project Overview

Nexus MCP is a Python 3.13+ MCP server that enables AI models to invoke agent
runners and integrations (Claude Code, Codex, OpenCode, OpenCode server) as tools.
Built with FastMCP 3.1+.

OpenCode Gemini-family model names are OpenCode provider/model configuration, not
support for a separate command-line runner.

## Build, Lint, Test Commands

```bash
# Setup (first time)
uv sync
uv run pre-commit install

# Run all tests (unit + e2e, fast)
uv run pytest

# Run single test file
uv run pytest tests/unit/test_types.py

# Run single test function
uv run pytest tests/unit/test_types.py::test_prompt_request_valid

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
- Pin reusable `shared-workflows` callers to immutable commit SHAs with a
  version comment, but keep PyPI publishing inline unless PyPI explicitly
  supports cross-repo reusable workflows for Trusted Publishing.

## Python Code Style

### Modern Syntax (Python 3.13+)

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
import asyncio
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field

from nexus_mcp.exceptions import SubprocessError
from nexus_mcp.types import AgentResponse
```

Public modules use `__all__` for re-exports.

## Types and Models

- Use Pydantic `BaseModel` for request/response types.
- Use `frozen=True` for immutable models.
- Define type aliases in `types.py`.
- Use `Protocol` for callback interfaces.

## Error Handling

Use exceptions from `exceptions.py`:

- `NexusMCPError`
- `SubprocessError`
- `SubprocessTimeoutError`
- `RetryableError`
- `ParseError`
- `CLINotFoundError`
- `UnsupportedAgentError`
- `ConfigurationError`

Include context in error messages and let exceptions propagate to the MCP boundary.

## Testing

- `tests/unit/` - fast, isolated unit tests
- `tests/e2e/` - in-process MCP protocol tests via `Client(mcp)`
- `tests/integration/` - slow, real CLI calls
- `tests/fixtures.py` - shared test factories

Test files mirror source modules as `test_<module_name>.py`.

Mock at the subprocess boundary (`asyncio.create_subprocess_exec`), not runner methods, unless
testing server-level orchestration.

Use factory helpers from `tests/fixtures.py`:

```python
req = make_prompt_request()
resp = make_agent_response(output="custom")
task = make_agent_task(cli="codex", prompt="X")
```

`asyncio_mode = "auto"` is configured, so async tests do not need
`@pytest.mark.asyncio`.

Run targeted tests after code changes when practical.

## Architecture Patterns

- Template Method: `AbstractRunner._execute()` defines the execution skeleton.
- Strategy: execution modes vary runner behavior.
- Factory: `RunnerFactory` creates runner instances.
- Chain of Responsibility: parser fallback chain handles output variants.

## Module Responsibilities

- `server.py` - FastMCP tool definitions, MCP boundary
- `types.py` - Pydantic models, type aliases
- `exceptions.py` - exception hierarchy
- `process.py` - subprocess wrapper
- `parser.py` - output parsing helpers
- `config.py` / `config_resolver.py` - configuration resolution
- `cli_detector.py` - runtime CLI detection
- `runners/` - CLI-specific implementations

## Search

- Prefer `rg` for text search and `rg --files` for file listing.
- Use `ast-grep --lang [language] -p '<pattern>'` for syntax-aware code search.

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`. Run manually:

```bash
uv run pre-commit run --all-files
```
