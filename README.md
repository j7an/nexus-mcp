# Nexus MCP

An MCP server that enables AI models to invoke AI CLI agents (Gemini CLI, Codex, Claude Code) as
tools. Provides parallel execution, automatic retries with exponential backoff, JSON-first response
parsing, and structured output through three MCP tools.

## Features

- **Parallel execution** — `batch_prompt` fans out tasks with `asyncio.gather` and a configurable
  semaphore (default concurrency: 3)
- **Automatic retries** — exponential backoff with full jitter for transient errors (HTTP 429/503)
- **Output handling** — JSON-first parsing, brace-depth fallback for noisy stdout, temp-file
  spillover for outputs exceeding 50 KB
- **Execution modes** — `default` (safe), `sandbox` (restricted), `yolo` (full auto-approve)
- **CLI detection** — auto-detects binary path, version, and JSON output capability at startup
- **Extensible** — implement `build_command` + `parse_output`, register in `RunnerFactory`

| Agent | Status |
|-------|--------|
| Gemini CLI | Supported |
| Codex | Planned |
| Claude Code | Planned |

## MCP Tools

All prompt tools run as background tasks — they return a task ID immediately so the client can
poll for results, preventing MCP timeouts for long operations (e.g. YOLO mode: 2–5 minutes).

| Tool | Task? | Description |
|------|-------|-------------|
| `batch_prompt` | Yes | Fan out prompts to multiple agents in parallel; returns `MultiPromptResponse` |
| `prompt` | Yes | Single-agent convenience wrapper; routes to `batch_prompt` |
| `list_agents` | No | Returns list of supported agent names |

## Quick Start

### Prerequisites

**Required:**
- **Python 3.13+** ([download](https://www.python.org/downloads/))
- **uv** dependency manager ([install guide](https://github.com/astral-sh/uv))
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv --version  # Verify installation
  ```

**Optional (for integration tests):**
- **Gemini CLI** v0.6.0+ — `npm install -g @google/gemini-cli`
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

# 5. Run the MCP server
uv run python -m nexus_mcp
```

## Configuration

### Global Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUS_OUTPUT_LIMIT_BYTES` | `50000` | Max output size in bytes before temp-file spillover |
| `NEXUS_TIMEOUT_SECONDS` | `600` | Subprocess timeout in seconds (10 minutes) |
| `NEXUS_RETRY_MAX_ATTEMPTS` | `3` | Max attempts including the first (set to 1 to disable retries) |
| `NEXUS_RETRY_BASE_DELAY` | `2.0` | Base seconds for exponential backoff |
| `NEXUS_RETRY_MAX_DELAY` | `60.0` | Maximum seconds to wait between retries |

### Agent-Specific Environment Variables

Pattern: `NEXUS_{AGENT}_{KEY}` (agent name uppercased)

| Variable | Description |
|----------|-------------|
| `NEXUS_GEMINI_PATH` | Override Gemini CLI binary path |
| `NEXUS_GEMINI_MODEL` | Default Gemini model (e.g. `gemini-2.5-flash`) |

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
├── src/nexus_mcp/
│   ├── __main__.py         # Entry point
│   ├── server.py           # FastMCP server + tools
│   ├── types.py            # Pydantic models
│   ├── exceptions.py       # Exception hierarchy
│   ├── config.py           # Environment variable config
│   ├── process.py          # Subprocess wrapper
│   ├── parser.py           # JSON→text fallback parsing
│   ├── cli_detector.py     # CLI binary detection + version checks
│   └── runners/
│       ├── base.py         # Protocol + ABC
│       ├── factory.py      # RunnerFactory
│       └── gemini.py       # GeminiRunner
├── tests/
│   ├── unit/               # Fast, mocked tests
│   ├── integration/        # Real CLI tests
│   └── fixtures.py         # Shared test utilities
├── .github/
│   └── workflows/          # CI, security, dependabot
├── pyproject.toml          # Dependencies + tool config
└── .pre-commit-config.yaml # Git hooks configuration
```

## Common Commands

```bash
# Start MCP server
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

- **Python 3.13+** required for modern syntax:
  - `type` keyword for type aliases: `type AgentName = str`
  - Union syntax: `str | None` (not `Optional[str]`)
  - `match` statements for complex conditionals
  - **NO** `from __future__ import annotations`

## Tool Configuration

- **Ruff:** line length 100, 17 rule sets (E/F/I/W + UP/FA/B/C4/SIM/RET/ICN/TID/TC/ISC/PTH/TD/NPY) — `pyproject.toml → [tool.ruff]`
- **Mypy:** strict mode, all type annotations required — `pyproject.toml → [tool.mypy]`
- **Pytest:** `asyncio_mode = "auto"`, no `@pytest.mark.asyncio` needed — `pyproject.toml → [tool.pytest.ini_options]`
- **Pre-commit:** ruff-check, ruff-format, mypy, trailing-whitespace, end-of-file-fixer — `.pre-commit-config.yaml`

## License

MIT
