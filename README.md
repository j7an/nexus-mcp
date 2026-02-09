# Nexus MCP

> **Work in progress / Non-functional**

A Model Context Protocol (MCP) server that enables AI models to invoke AI CLI agents (Gemini CLI, Codex, Claude Code) as tools. Provides structured prompting and response handling through MCP tools with JSON-first parsing and text fallback strategies.

## Quick Start

### Prerequisites

**Required:**
- **Python 3.12+** ([download](https://www.python.org/downloads/))
- **uv** dependency manager ([install guide](https://github.com/astral-sh/uv))
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv --version  # Verify installation
  ```

**Optional (for integration tests):**
- **Gemini CLI** v0.12.0+ — `npm install -g @google/gemini-cli`
- **Codex** — check with `codex --version`
- **Claude Code** — check with `claude --version`

> **Note:** Integration tests are optional. Unit tests run without CLI dependencies via subprocess mocking.

### Setup for Development

```bash
# 1. Clone the repository
git clone <repository-url>
cd nexus-mcp

# 2. Install dependencies
uv sync

# 3. Install pre-commit hooks (runs linting/formatting on commit)
uv run pre-commit install

# 4. Verify installation
uv run pytest                    # Run tests
uv run mypy src/nexus_mcp        # Type checking
uv run ruff check .              # Linting

# 5. Run the MCP server (after implementation)
uv run python -m nexus_mcp
```

## Development Workflow

### Adding Dependencies

```bash
# Production dependencies
uv add fastmcp pydantic

# Development dependencies
uv add --dev pytest pytest-asyncio mypy ruff

# Sync environment after changes
uv sync
```

### Code Quality

All quality checks run automatically via pre-commit hooks. Run manually:

```bash
# Lint and format
uv run ruff check .              # Check for issues
uv run ruff check --fix .        # Auto-fix issues
uv run ruff format .             # Format code

# Type checking (strict mode)
uv run mypy src/nexus_mcp

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

### Testing

This project follows **Test-Driven Development (TDD)** with strict Red→Green→Refactor cycles.

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=nexus_mcp --cov-report=term-missing

# Run specific test types
uv run pytest -m integration           # Integration tests (requires CLIs)
uv run pytest -m "not integration"     # Unit tests only
uv run pytest -m "not slow"            # Skip slow tests

# Run specific test file
uv run pytest tests/unit/runners/test_gemini.py
```

**Test markers:**
- `@pytest.mark.integration` — requires real CLI installations
- `@pytest.mark.slow` — tests taking >1 second

### Project Structure

```
nexus-mcp/
├── src/nexus_mcp/          # Main package (implementation pending)
│   ├── server.py           # FastMCP server + tools
│   ├── types.py            # Pydantic models
│   ├── exceptions.py       # Exception hierarchy
│   └── runners/            # CLI agent runners
│       ├── base.py         # Protocol + ABC
│       ├── factory.py      # RunnerFactory
│       └── gemini.py       # GeminiRunner
├── tests/
│   ├── unit/               # Fast, mocked tests
│   ├── integration/        # Real CLI tests (future)
│   └── fixtures.py         # Shared test utilities
├── pyproject.toml          # Dependencies + tool config
└── .pre-commit-config.yaml # Git hooks configuration
```

## Common Commands

```bash
# Start MCP server (after implementation)
uv run python -m nexus_mcp

# Run TDD cycle
uv run pytest --cov=nexus_mcp -v

# Code quality checks
uv run ruff check . && uv run ruff format .
uv run mypy src/nexus_mcp

# Pre-commit hooks
uv run pre-commit run --all-files
```

## Python Requirements

- **Python 3.12+** required for modern syntax:
  - `type` keyword for type aliases: `type AgentName = str`
  - Union syntax: `str | None` (not `Optional[str]`)
  - `match` statements for complex conditionals
  - **NO** `from __future__ import annotations`

## Configuration

### Ruff (Linter + Formatter)
- Line length: 100 characters
- 17 rule sets enabled (E/F/I/W core + UP/FA/B/C4/SIM/RET/ICN/TID/TC/ISC/PTH/TD/NPY)
- Config: `pyproject.toml` → `[tool.ruff]`

### Mypy (Type Checker)
- Strict mode enabled
- All type annotations required
- Config: `pyproject.toml` → `[tool.mypy]`

### Pytest
- `pytest-asyncio>=1.1.0` with `asyncio_mode = "auto"`
- No `@pytest.mark.asyncio` decorators needed
- Config: `pyproject.toml` → `[tool.pytest.ini_options]`

### Pre-commit Hooks
- ruff-check, ruff-format, mypy, trailing-whitespace, end-of-file-fixer
- Config: `.pre-commit-config.yaml`

## License

MIT
