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
parsing, and structured output through six MCP tools.

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
- **Session preferences** — set defaults for execution mode, model, max retries, output limit, and timeout once per session; subsequent calls inherit them without repeating parameters
- **Tool timeouts** — configurable safety timeout (default 15 min) cancels long-running tool calls to prevent the server from blocking indefinitely
- **Extensible** — implement `build_command` + `parse_output`, register in `RunnerFactory`

| Agent | Status |
|-------|--------|
| Gemini CLI | Supported |
| Codex | Supported |
| Claude Code | Supported |
| OpenCode | Supported |

## Installation

### Run with uvx (recommended)

```bash
uvx nexus-mcp
```

`uvx` installs the package in an ephemeral virtual environment and runs it — no cloning required.

To check the installed version:

```bash
uvx nexus-mcp --version
```

To update to the latest version:

```bash
uvx --reinstall nexus-mcp
```

### MCP Client Configuration

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "nexus-mcp": {
      "command": "uvx",
      "args": ["nexus-mcp"],
      "env": {
        "NEXUS_GEMINI_MODEL": "gemini-2.5-flash",
        "NEXUS_GEMINI_MODELS": "gemini-2.5-pro,gemini-2.5-flash,gemini-2.0-flash",
        "NEXUS_CODEX_MODEL": "gpt-5.2",
        "NEXUS_CODEX_MODELS": "gpt-5.4,gpt-5.4-mini,gpt-5.3-codex,gpt-5.2-codex,gpt-5.2,gpt-5.1-codex-max,gpt-5.1-codex-mini",
        "NEXUS_OPENCODE_MODEL": "ollama-cloud/kimi-k2.5",
        "NEXUS_OPENCODE_MODELS": "ollama-cloud/glm-5,ollama-cloud/kimi-k2.5,ollama-cloud/qwen3-coder-next,ollama-cloud/minimax-m2.5,ollama/gemini-3-flash-preview"
      }
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
      "args": ["nexus-mcp"],
      "env": {
        "NEXUS_GEMINI_MODEL": "gemini-2.5-flash",
        "NEXUS_GEMINI_MODELS": "gemini-2.5-pro,gemini-2.5-flash,gemini-2.0-flash",
        "NEXUS_CODEX_MODEL": "gpt-5.2",
        "NEXUS_CODEX_MODELS": "gpt-5.4,gpt-5.4-mini,gpt-5.3-codex,gpt-5.2-codex,gpt-5.2,gpt-5.1-codex-max,gpt-5.1-codex-mini",
        "NEXUS_OPENCODE_MODEL": "ollama-cloud/kimi-k2.5",
        "NEXUS_OPENCODE_MODELS": "ollama-cloud/glm-5,ollama-cloud/kimi-k2.5,ollama-cloud/qwen3-coder-next,ollama-cloud/minimax-m2.5,ollama/gemini-3-flash-preview"
      }
    }
  }
}
```

**Claude Code** (CLI):

```bash
claude mcp add nexus-mcp \
  -e NEXUS_GEMINI_MODEL=gemini-2.5-flash \
  -e NEXUS_GEMINI_MODELS=gemini-2.5-pro,gemini-2.5-flash,gemini-2.0-flash \
  -e NEXUS_CODEX_MODEL=gpt-5.2 \
  -e NEXUS_CODEX_MODELS=gpt-5.4,gpt-5.4-mini,gpt-5.3-codex,gpt-5.2-codex,gpt-5.2,gpt-5.1-codex-max,gpt-5.1-codex-mini \
  -e NEXUS_OPENCODE_MODEL=ollama-cloud/kimi-k2.5 \
  -e NEXUS_OPENCODE_MODELS=ollama-cloud/glm-5,ollama-cloud/kimi-k2.5,ollama-cloud/qwen3-coder-next,ollama-cloud/minimax-m2.5,ollama/gemini-3-flash-preview \
  -- uvx nexus-mcp
```

**Generic stdio config** (any MCP-compatible client):

```json
{
  "command": "uvx",
  "args": ["nexus-mcp"],
  "transport": "stdio",
  "env": {
    "NEXUS_GEMINI_MODEL": "gemini-2.5-flash",
    "NEXUS_CODEX_MODEL": "gpt-5.2",
    "NEXUS_OPENCODE_MODEL": "ollama-cloud/kimi-k2.5"
  }
}
```

All `env` keys are optional — see [Configuration](#configuration) for the full list.

### Setup for Development

**Prerequisites:**
- **Python 3.13+** ([download](https://www.python.org/downloads/))
- **uv** dependency manager ([install guide](https://github.com/astral-sh/uv))
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

**Optional (for integration tests):**
- **Gemini CLI** v0.6.0+ — `npm install -g @google/gemini-cli`
- **Codex** — check with `codex --version`
- **Claude Code** — check with `claude --version`
- **OpenCode** — check with `opencode --version`

> **Note:** Integration tests are optional. Unit tests run without CLI dependencies via subprocess mocking.

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

## Usage

Once nexus-mcp is configured in your MCP client, your AI assistant automatically sees its tools.
The reliable trigger is **explicitly asking for output from an external AI agent** (e.g. Gemini, Codex, Claude Code, OpenCode).
Generic "do this in parallel" prompts may be handled by the host AI's own capabilities instead.
Because `cli` is a required parameter, the assistant typically calls `list_runners` first to discover
what's available, then fans out your request accordingly.

### Fan out a research question (batch_prompt)

**You say to your AI assistant:**
> "Get perspectives from Gemini, Codex, and OpenCode on transformer architectures — summary, limitations, and applications."

**Your AI assistant first calls `list_runners` to discover available runners:**

```json
{}
```

**Response:** structured metadata for each runner (models, available, execution_modes, defaults)

**Then calls `batch_prompt` with the discovered runners:**

```json
{
  "tasks": [
    { "cli": "gemini", "prompt": "Summarize the key findings of the Attention Is All You Need paper", "label": "gemini-summary" },
    { "cli": "codex", "prompt": "What are the main limitations of transformer architectures?", "label": "codex-limitations" },
    { "cli": "opencode", "prompt": "List 3 real-world applications of transformers beyond NLP", "label": "opencode-applications" }
  ]
}
```

Runner discovery happens once per session; subsequent examples skip the `list_runners` step.

### Code review from multiple angles (batch_prompt)

**You say to your AI assistant:**
> "Have Gemini, Codex, and OpenCode each review this diff in parallel — I want three independent perspectives."

**Your AI assistant calls `batch_prompt`:**

```json
{
  "tasks": [
    { "cli": "gemini", "prompt": "Review this diff for security vulnerabilities and logic errors:\n\n<paste diff>", "label": "gemini-review" },
    { "cli": "codex", "prompt": "Review this diff for correctness and edge cases:\n\n<paste diff>", "label": "codex-review" },
    { "cli": "opencode", "prompt": "Review this diff for style and maintainability:\n\n<paste diff>", "label": "opencode-review" }
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

**Or target Codex:**

```json
{
  "cli": "codex",
  "prompt": "Explain the difference between TCP and UDP in simple terms",
  "model": "gpt-5.2"
}
```

**Or OpenCode:**

```json
{
  "cli": "opencode",
  "prompt": "Explain the difference between TCP and UDP in simple terms",
  "model": "ollama-cloud/kimi-k2.5"
}
```

### Session preferences (set_preferences)

**You say to your AI assistant:**
> "For the rest of this session, use YOLO mode with Gemini Flash — I don't want to repeat those settings on every call."

**Your AI assistant calls `set_preferences` once:**

```json
{
  "execution_mode": "yolo",
  "model": "gemini-2.5-flash",
  "max_retries": 5
}
```

**Response:**
```
Preferences set: {"execution_mode": "yolo", "model": "gemini-2.5-flash", "max_retries": 5, "output_limit": null, "timeout": null, "retry_base_delay": null, "retry_max_delay": null}
```

**Subsequent `prompt` and `batch_prompt` calls omit those fields — they inherit from the session:**

```json
{
  "cli": "gemini",
  "prompt": "Summarize the latest developments in Rust's async ecosystem"
}
```

The fallback chain is: **explicit parameter → session preference → per-runner env → global env → hardcoded default**.
To override for one call, pass the parameter directly — it takes precedence without changing the session.
To clear a single preference, use `set_preferences` with the corresponding `clear_*` flag (e.g. `clear_execution_mode: true`, `clear_model: true`, `clear_max_retries: true`, `clear_output_limit: true`, `clear_timeout: true`, `clear_retry_base_delay: true`, `clear_retry_max_delay: true`).

## MCP Tools

All prompt tools run as background tasks — they return a task ID immediately so the client can
poll for results, preventing MCP timeouts for long operations (e.g. YOLO mode: 2–5 minutes).

| Tool | Task? | Description |
|------|-------|-------------|
| `batch_prompt` | Yes | Fan out prompts to multiple runners in parallel; returns `MultiPromptResponse` |
| `prompt` | Yes | Single-runner convenience wrapper; routes to `batch_prompt` |
| `list_runners` | No | Returns structured metadata for each runner (models, available, execution_modes, defaults) |
| `set_preferences` | No | Set or selectively clear session defaults for execution mode, model, max retries, output limit, timeout, retry base delay, and retry max delay |
| `get_preferences` | No | Retrieve current session preferences |
| `clear_preferences` | No | Reset all session preferences |

### `batch_prompt`

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
| `max_retries` | No | session pref or env default | Max retry attempts for transient errors |
| `output_limit` | No | session pref or env default | Max output bytes before temp-file spillover |
| `timeout` | No | session pref or env default | Subprocess timeout in seconds |
| `retry_base_delay` | No | session pref or env default | Base delay seconds for exponential backoff |
| `retry_max_delay` | No | session pref or env default | Max delay cap for backoff in seconds |

### `prompt`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `cli` | Yes | — | Runner name |
| `prompt` | Yes | — | Prompt text |
| `context` | No | `{}` | Optional context metadata dict |
| `execution_mode` | No | session pref or `"default"` | `"default"` or `"yolo"` |
| `model` | No | session pref or CLI default | Model name override |
| `max_retries` | No | session pref or env default | Max retry attempts for transient errors |
| `output_limit` | No | session pref or env default | Max output bytes before temp-file spillover |
| `timeout` | No | session pref or env default | Subprocess timeout in seconds |
| `retry_base_delay` | No | session pref or env default | Base delay seconds for exponential backoff |
| `retry_max_delay` | No | session pref or env default | Max delay cap for backoff in seconds |

### `list_runners`

No parameters. Returns a list of `RunnerInfo` objects with fields: `name`, `models`, `available`, `execution_modes`, `defaults` (nested `OperationalDefaults` with `model`, `timeout`, `output_limit`, `max_retries`, etc.).

### `set_preferences`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `execution_mode` | No | — | `"default"` or `"yolo"` |
| `model` | No | — | Model name (e.g. `"gemini-2.5-flash"`) |
| `max_retries` | No | — | Max total attempts including the first (≥1; 1 means run once, no retries) |
| `output_limit` | No | — | Max output bytes before temp-file spillover (≥1) |
| `timeout` | No | — | Subprocess timeout in seconds (≥1) |
| `retry_base_delay` | No | — | Base delay seconds for exponential backoff (≥0) |
| `retry_max_delay` | No | — | Max delay cap for backoff in seconds (≥0) |
| `clear_execution_mode` | No | `false` | Clear execution mode (takes precedence if `execution_mode` is also provided) |
| `clear_model` | No | `false` | Clear model (takes precedence if `model` is also provided) |
| `clear_max_retries` | No | `false` | Clear max retries (takes precedence if `max_retries` is also provided) |
| `clear_output_limit` | No | `false` | Clear output limit (takes precedence if `output_limit` is also provided) |
| `clear_timeout` | No | `false` | Clear timeout (takes precedence if `timeout` is also provided) |
| `clear_retry_base_delay` | No | `false` | Clear retry base delay |
| `clear_retry_max_delay` | No | `false` | Clear retry max delay |

### `get_preferences`

No parameters. Returns a dict with `execution_mode`, `model`, `max_retries`, `output_limit`, `timeout`, `retry_base_delay`, and `retry_max_delay` keys (`null` when unset).

### `clear_preferences`

No parameters. Resets all session preferences.

### Managing Session Preferences

| Operation | Tool | Notes |
|-----------|------|-------|
| Set one or more fields | `set_preferences` | Pass only the fields you want to change |
| Read current values | `get_preferences` | Returns `{execution_mode, model, max_retries, output_limit, timeout, retry_base_delay, retry_max_delay}` with `null` for unset fields |
| Clear all fields | `clear_preferences` | Reverts to per-call defaults |
| Clear one preference | `set_preferences` with `clear_model: true`, `clear_execution_mode: true`, `clear_max_retries: true`, `clear_output_limit: true`, `clear_timeout: true`, `clear_retry_base_delay: true`, or `clear_retry_max_delay: true` | Other preferences are preserved |

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
| `NEXUS_CLI_DETECTION_TIMEOUT` | `30` | Timeout in seconds for CLI binary version detection at startup |
| `NEXUS_EXECUTION_MODE` | `default` | Global execution mode (`default` or `yolo`) |

### Per-Runner Environment Variables

Pattern: `NEXUS_{AGENT}_{KEY}` (agent name uppercased). Per-runner values override global values.

Valid `{AGENT}` values: `CLAUDE`, `CODEX`, `GEMINI`, `OPENCODE`

| Variable pattern | Example | Description |
|----------|---------|-------------|
| `NEXUS_{AGENT}_MODEL` | `NEXUS_GEMINI_MODEL=gemini-2.5-flash` | Default model for this runner |
| `NEXUS_{AGENT}_MODELS` | `NEXUS_GEMINI_MODELS=gemini-2.5-flash,gemini-2.5-pro` | Comma-separated model list (surfaced in `list_runners`) |
| `NEXUS_{AGENT}_TIMEOUT` | `NEXUS_GEMINI_TIMEOUT=900` | Subprocess timeout override |
| `NEXUS_{AGENT}_OUTPUT_LIMIT` | `NEXUS_CODEX_OUTPUT_LIMIT=100000` | Output limit override |
| `NEXUS_{AGENT}_MAX_RETRIES` | `NEXUS_CLAUDE_MAX_RETRIES=5` | Max retry attempts override |
| `NEXUS_{AGENT}_RETRY_BASE_DELAY` | `NEXUS_GEMINI_RETRY_BASE_DELAY=1.0` | Backoff base delay override |
| `NEXUS_{AGENT}_RETRY_MAX_DELAY` | `NEXUS_GEMINI_RETRY_MAX_DELAY=30.0` | Backoff max delay override |
| `NEXUS_{AGENT}_EXECUTION_MODE` | `NEXUS_GEMINI_EXECUTION_MODE=yolo` | Execution mode override |

Invalid per-runner values are silently ignored (the global or hardcoded default is used instead).

## Development

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

### Adding Dependencies

```bash
uv add <package>              # Production dependency
uv add --dev <package>        # Development dependency
uv sync                       # Sync environment after changes
```

### Tool Configuration

- **Ruff:** line length 100, 17 rule sets (E/F/I/W + UP/FA/B/C4/SIM/RET/ICN/TID/TC/ISC/PTH/TD/NPY) — `pyproject.toml → [tool.ruff]`
- **Mypy:** strict mode, all type annotations required — `pyproject.toml → [tool.mypy]`
- **Pytest:** `asyncio_mode = "auto"`, no `@pytest.mark.asyncio` needed — `pyproject.toml → [tool.pytest.ini_options]`
- **Pre-commit:** ruff-check, ruff-format, mypy, trailing-whitespace, end-of-file-fixer — `.pre-commit-config.yaml`

### Python 3.13+ Syntax

- `type` keyword for type aliases: `type AgentName = str`
- Union syntax: `str | None` (not `Optional[str]`)
- `match` statements for complex conditionals
- **NO** `from __future__ import annotations`

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
│   ├── e2e/                # End-to-end MCP protocol tests
│   ├── integration/        # Real CLI tests
│   └── fixtures.py         # Shared test utilities
├── .github/
│   └── workflows/          # CI, security, dependabot
├── pyproject.toml          # Dependencies + tool config
└── .pre-commit-config.yaml # Git hooks configuration
```

## License

MIT
