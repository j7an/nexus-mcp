# Nexus MCP

[![PyPI](https://img.shields.io/pypi/v/nexus-mcp)](https://pypi.org/project/nexus-mcp/)
[![Python 3.13+](https://img.shields.io/pypi/pyversions/nexus-mcp)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple)](https://modelcontextprotocol.io/)

A MCP server that enables AI models to invoke AI CLI agents (Gemini CLI, Codex, Claude Code, OpenCode) as
tools. Provides parallel execution, automatic retries with exponential backoff, JSON-first response
parsing, and structured output through three MCP tools.

## Use Cases

Nexus MCP is useful whenever a task benefits from querying multiple AI agents in
parallel rather than sequentially:

- **Research & summarization** — fan out a topic to multiple agents, then
  synthesize their responses into a single summary with diverse perspectives
- **Code review** — send different files or review angles (security, correctness,
  style) to separate agents simultaneously
- **Multi-model comparison** — prompt the same question to different models and
  compare outputs side-by-side for quality or consistency
- **Bulk content generation** — generate multiple test cases, translations, or
  documentation pages concurrently instead of one at a time
- **Second-opinion workflows** — get independent answers from separate agents
  before making a decision, reducing single-model bias

## Features

- **Parallel execution** — `batch_prompt` fans out tasks with `asyncio.gather` and a configurable
  semaphore (default concurrency: 3)
- **Automatic retries** — exponential backoff with full jitter for transient errors (HTTP 429/503)
- **Output handling** — JSON-first parsing, brace-depth fallback for noisy stdout, temp-file
  spillover for outputs exceeding 50 KB
- **Execution modes** — `default` (safe, no auto-approve), `yolo` (full auto-approve)
- **CLI detection** — auto-detects binary path, version, and JSON output capability at startup
- **Session preferences** — set defaults for execution mode and model once per session; subsequent calls inherit them without repeating parameters
- **Tool timeouts** — configurable safety timeout (default 15 min) cancels long-running tool calls to prevent the server from blocking indefinitely
- **Extensible** — implement `build_command` + `parse_output`, register in `RunnerFactory`

| Agent | Status |
|-------|--------|
| Gemini CLI | Supported |
| Codex | Supported |
| Claude Code | Supported |
| OpenCode | Supported |

## Usage

Once nexus-mcp is configured in your MCP client, your AI assistant automatically sees its tools.
The reliable trigger is **explicitly asking for output from an external AI agent** (e.g. Gemini, Codex, Claude Code, OpenCode).
Generic "do this in parallel" prompts may be handled by the host AI's own capabilities instead.
Because `cli` is a required parameter, the assistant typically calls `list_runners` first to discover
what's available, then fans out your request accordingly.

### Fan out a research question (batch_prompt)

**You say to your AI assistant:**
> "Get Gemini's perspective on transformer architectures — I want its summary of the Attention Is All You Need paper, its view on the main limitations, and its list of real-world applications beyond NLP."

**Your AI assistant first calls `list_runners` to discover available runners:**

```json
{}
```

**Response:** structured metadata for each runner (provider, models, available, execution_modes, default_model)

**Then calls `batch_prompt` with the discovered runner:**

```json
{
  "tasks": [
    { "cli": "gemini", "prompt": "Summarize the key findings of the Attention Is All You Need paper", "label": "summary" },
    { "cli": "gemini", "prompt": "What are the main limitations of transformer architectures?", "label": "limitations" },
    { "cli": "gemini", "prompt": "List 3 real-world applications of transformers beyond NLP", "label": "applications" }
  ]
}
```

Runner discovery happens once per session; subsequent examples skip the `list_runners` step.

### Code review from multiple angles (batch_prompt)

**You say to your AI assistant:**
> "Have Gemini review this diff from three angles in parallel: security vulnerabilities, logic errors, and style issues."

**Your AI assistant calls `batch_prompt`:**

```json
{
  "tasks": [
    { "cli": "gemini", "prompt": "Review this diff for security vulnerabilities:\n\n<paste diff>", "label": "security" },
    { "cli": "gemini", "prompt": "Review this diff for correctness and logic errors:\n\n<paste diff>", "label": "correctness" },
    { "cli": "gemini", "prompt": "Review this diff for style and maintainability:\n\n<paste diff>", "label": "style" }
  ]
}
```

### Single-agent prompt (prompt)

**You say to your AI assistant:**
> "Ask Gemini Flash to explain the difference between TCP and UDP in simple terms."

**Your AI assistant calls `prompt`:**

```json
{
  "cli": "gemini",
  "prompt": "Explain the difference between TCP and UDP in simple terms",
  "model": "gemini-2.5-flash"
}
```

### Session preferences (set_preferences)

**You say to your AI assistant:**
> "For the rest of this session, use YOLO mode with Gemini Flash — I don't want to repeat those settings on every call."

**Your AI assistant calls `set_preferences` once:**

```json
{
  "execution_mode": "yolo",
  "model": "gemini-2.5-flash"
}
```

**Response:**
```
Preferences set: {"execution_mode": "yolo", "model": "gemini-2.5-flash"}
```

**Subsequent `prompt` and `batch_prompt` calls omit those fields — they inherit from the session:**

```json
{
  "cli": "gemini",
  "prompt": "Summarize the latest developments in Rust's async ecosystem"
}
```

The fallback chain is: **explicit parameter → session preference → system default**.
To override for one call, pass the parameter directly — it takes precedence without changing the session.
To clear a single preference, use `set_preferences` with the corresponding `clear_*` flag (e.g. `clear_model: true`).

### Managing Session Preferences

| Operation | Tool | Notes |
|-----------|------|-------|
| Set one or both fields | `set_preferences` | Pass only the fields you want to change |
| Read current values | `get_preferences` | Returns `{execution_mode, model}` with `null` for unset fields |
| Clear all fields | `clear_preferences` | Reverts to per-call defaults |
| Clear one preference | `set_preferences` with `clear_model: true` or `clear_execution_mode: true` | Other preference is preserved |

### Parameter Reference

#### `batch_prompt`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `tasks` | Yes | — | List of task objects (see below) |
| `max_concurrency` | No | `3` | Max parallel agent invocations |

**Task object fields:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `cli` | Yes | — | Runner name (e.g. `"gemini"`) |
| `prompt` | Yes | — | Prompt text |
| `label` | No | auto | Display label for results (auto-assigned from runner name if omitted) |
| `context` | No | `{}` | Optional context metadata dict |
| `execution_mode` | No | session pref or `"default"` | `"default"` or `"yolo"` |
| `model` | No | session pref or CLI default | Model name override |
| `max_retries` | No | env default | Max retry attempts for transient errors |

#### `prompt`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `cli` | Yes | — | Runner name |
| `prompt` | Yes | — | Prompt text |
| `context` | No | `{}` | Optional context metadata dict |
| `execution_mode` | No | session pref or `"default"` | `"default"` or `"yolo"` |
| `model` | No | session pref or CLI default | Model name override |
| `max_retries` | No | env default | Max retry attempts for transient errors |

#### `list_runners`

No parameters. Returns a list of `RunnerInfo` objects with fields: `name`, `provider`, `models`, `available`, `execution_modes`, `default_model`.

#### `set_preferences`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `execution_mode` | No | — | `"default"` or `"yolo"` |
| `model` | No | — | Model name (e.g. `"gemini-2.5-flash"`) |
| `clear_execution_mode` | No | `false` | Clear execution mode (takes precedence if `execution_mode` is also provided) |
| `clear_model` | No | `false` | Clear model (takes precedence if `model` is also provided) |

#### `get_preferences`

No parameters. Returns a dict with `execution_mode` and `model` keys (`null` when unset).

#### `clear_preferences`

No parameters. Resets all session preferences.

## MCP Tools

All prompt tools run as background tasks — they return a task ID immediately so the client can
poll for results, preventing MCP timeouts for long operations (e.g. YOLO mode: 2–5 minutes).

| Tool | Task? | Description |
|------|-------|-------------|
| `batch_prompt` | Yes | Fan out prompts to multiple runners in parallel; returns `MultiPromptResponse` |
| `prompt` | Yes | Single-runner convenience wrapper; routes to `batch_prompt` |
| `list_runners` | No | Returns structured metadata for each runner (provider, models, available, execution_modes, default_model) |
| `set_preferences` | No | Set or selectively clear session defaults for execution mode and model |
| `get_preferences` | No | Retrieve current session preferences |
| `clear_preferences` | No | Reset all session preferences |

## Installation

### Run with uvx (recommended)

```bash
uvx nexus-mcp
```

`uvx` installs the package in an ephemeral virtual environment and runs it — no cloning required.

### MCP Client Configuration

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "nexus-mcp": {
      "command": "uvx",
      "args": ["nexus-mcp"]
    }
  }
}
```

**Cursor** (`.cursor/mcp.json` in your project or `~/.cursor/mcp.json` globally):

```json
{
  "mcpServers": {
    "nexus-mcp": {
      "command": "uvx",
      "args": ["nexus-mcp"]
    }
  }
}
```

**Claude Code** (CLI):

```bash
claude mcp add nexus-mcp uvx nexus-mcp
```

**Generic stdio config** (any MCP-compatible client):

```json
{
  "command": "uvx",
  "args": ["nexus-mcp"],
  "transport": "stdio"
}
```

> **Tip:** Pass environment variables (e.g. `NEXUS_GEMINI_MODEL`) via your client's `env` key.

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
- **OpenCode** — check with `opencode --version`

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
| `NEXUS_TOOL_TIMEOUT_SECONDS` | `900` | Tool-level timeout in seconds (15 minutes); set to `0` to disable |
| `NEXUS_RETRY_MAX_ATTEMPTS` | `3` | Max attempts including the first (set to 1 to disable retries) |
| `NEXUS_RETRY_BASE_DELAY` | `2.0` | Base seconds for exponential backoff |
| `NEXUS_RETRY_MAX_DELAY` | `60.0` | Maximum seconds to wait between retries |

### Agent-Specific Environment Variables

Pattern: `NEXUS_{AGENT}_{KEY}` (agent name uppercased)

| Variable | Description |
|----------|-------------|
| `NEXUS_CLAUDE_PATH` | Override Claude CLI binary path |
| `NEXUS_CLAUDE_MODEL` | Default Claude model (e.g. `claude-sonnet-4-6`) |
| `NEXUS_CODEX_PATH` | Override Codex CLI binary path |
| `NEXUS_CODEX_MODEL` | Default Codex model |
| `NEXUS_GEMINI_PATH` | Override Gemini CLI binary path |
| `NEXUS_GEMINI_MODEL` | Default Gemini model (e.g. `gemini-2.5-flash`) |
| `NEXUS_OPENCODE_PATH` | Override OpenCode CLI binary path |
| `NEXUS_OPENCODE_MODEL` | Default OpenCode model |

### Runner Config File (`nexus-mcp.toml`)

An optional TOML file provides provider and model metadata per runner. By default, nexus-mcp looks for `nexus-mcp.toml` in the current working directory. Override the path with `NEXUS_CONFIG_PATH`.

```toml
[runners.gemini]
provider = "Google"
models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

[runners.codex]
provider = "OpenAI"
models = ["o4-mini", "o3"]

[runners.claude]
provider = "Anthropic"
models = ["claude-sonnet-4-6", "claude-opus-4-5"]

[runners.opencode]
provider = "OpenCode"
models = []
```

All fields are optional. Omitting a runner section uses built-in defaults. The `models` list is surfaced in `list_runners` output so clients can discover available model names without hard-coding them.

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
│       ├── claude.py       # ClaudeRunner
│       ├── codex.py        # CodexRunner
│       ├── gemini.py       # GeminiRunner
│       └── opencode.py     # OpenCodeRunner
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
uvx nexus-mcp                    # Recommended (no clone needed)
uv run python -m nexus_mcp      # Development (from cloned repo)

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
