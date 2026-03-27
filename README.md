# Nexus MCP
<!-- mcp-name: io.github.j7an/nexus-mcp -->

[![PyPI](https://img.shields.io/pypi/v/nexus-mcp)](https://pypi.org/project/nexus-mcp/)
[![Python 3.13+](https://img.shields.io/pypi/pyversions/nexus-mcp)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple)](https://modelcontextprotocol.io/)

An MCP server that enables AI models to invoke AI CLI agents (Gemini CLI, Codex, Claude Code, OpenCode) as
tools. Provides parallel execution, automatic retries with exponential backoff, JSON-first response
parsing, 10 discoverable prompt templates, model tier classification, and persistent preferences
through seven MCP tools, four MCP resources, and ten MCP prompts.

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
- **Persistent preferences** — set defaults for execution mode, model, retries, output limit, and timeout; preferences persist across MCP sessions via the backing store (MemoryStore default, FileTreeStore/RedisStore for restart persistence)
- **Prompt templates** — 10 discoverable workflow scaffolds (code review, debug, research, implement feature, etc.) via `list_prompts`/`get_prompt`; each returns structured messages with expert framing the client can use or ignore
- **Model tier classification** — heuristic-based model classification into quick/standard/thorough tiers; clients can override with sampling or live benchmarks. The `nexus://runners` resource includes tier data per model
- **Tool timeouts** — configurable safety timeout (default 15 min) cancels long-running tool calls to prevent the server from blocking indefinitely
- **Client-visible logging** — runner events (retries, output truncation, error recovery) are sent to MCP clients via protocol notifications, not just server stderr
- **Elicitation** — interactive parameter resolution via MCP elicitation; disambiguates missing CLI, offers model selection, confirms YOLO mode, and prompts for elaboration on vague prompts. Auto-detects client support and skips gracefully when unavailable. Suppression flags prevent repeat prompts within a session
- **Benchmark data sources** — server instructions include URLs for Artificial Analysis, OpenRouter, Chatbot Arena, and LLM Stats so clients can fetch live model benchmarks without API keys
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

<details>
<summary><h3>MCP Client Configuration</h3></summary>

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "nexus-mcp": {
      "command": "uvx",
      "args": ["nexus-mcp"],
      "env": {
        "NEXUS_GEMINI_MODEL": "gemini-3-flash-preview",
        "NEXUS_GEMINI_MODELS": "gemini-3.1-pro-preview,gemini-3-flash-preview,gemini-2.5-pro,gemini-2.5-flash,gemini-2.5-flash-lite",
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
        "NEXUS_GEMINI_MODEL": "gemini-3-flash-preview",
        "NEXUS_GEMINI_MODELS": "gemini-3.1-pro-preview,gemini-3-flash-preview,gemini-2.5-pro,gemini-2.5-flash,gemini-2.5-flash-lite",
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
  -e NEXUS_GEMINI_MODEL=gemini-3-flash-preview \
  -e NEXUS_GEMINI_MODELS=gemini-3.1-pro-preview,gemini-3-flash-preview,gemini-2.5-pro,gemini-2.5-flash,gemini-2.5-flash-lite \
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
    "NEXUS_GEMINI_MODEL": "gemini-3-flash-preview",
    "NEXUS_CODEX_MODEL": "gpt-5.2",
    "NEXUS_OPENCODE_MODEL": "ollama-cloud/kimi-k2.5"
  }
}
```

All `env` keys are optional — see [Configuration](#configuration) for the full list.

</details>

<details>
<summary><h3>Setup for Development</h3></summary>

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

</details>

## Usage

Once nexus-mcp is configured in your MCP client, your AI assistant automatically sees its tools.
The reliable trigger is **explicitly asking for output from an external AI agent** (e.g. Gemini, Codex, Claude Code, OpenCode).
Generic "do this in parallel" prompts may be handled by the host AI's own capabilities instead.
The `cli` parameter is optional — if omitted and the client supports MCP elicitation, the server will
ask which runner to use. The server provides runner metadata (names, models, availability,
execution modes) in its connection instructions — no discovery call needed. The `cli` parameter
includes a JSON schema enum listing valid runner names.

<details>
<summary><h3>Usage Examples</h3></summary>

#### Fan out a research question (batch_prompt)

**You say:** "Get perspectives from Gemini, Codex, and OpenCode on transformer architectures."

```json
{
  "tasks": [
    { "cli": "gemini", "prompt": "Summarize the key findings of the Attention Is All You Need paper", "label": "gemini-summary" },
    { "cli": "codex", "prompt": "What are the main limitations of transformer architectures?", "label": "codex-limitations" },
    { "cli": "opencode", "prompt": "List 3 real-world applications of transformers beyond NLP", "label": "opencode-applications" }
  ]
}
```

#### Code review from multiple angles (batch_prompt)

**You say:** "Have Gemini, Codex, and OpenCode each review this diff in parallel."

```json
{
  "tasks": [
    { "cli": "gemini", "prompt": "Review this diff for security vulnerabilities:\n\n<paste diff>", "label": "gemini-review" },
    { "cli": "codex", "prompt": "Review this diff for correctness and edge cases:\n\n<paste diff>", "label": "codex-review" },
    { "cli": "opencode", "prompt": "Review this diff for style and maintainability:\n\n<paste diff>", "label": "opencode-review" }
  ]
}
```

#### Single-agent prompt

**You say:** "Ask Gemini Flash to explain the difference between TCP and UDP."

```json
{ "cli": "gemini", "prompt": "Explain the difference between TCP and UDP in simple terms", "model": "gemini-3-flash-preview" }
```

#### Elicitation (server picks the runner)

**You say:** "Explain the CAP theorem using one of the available agents."

```json
{ "prompt": "Explain the CAP theorem in simple terms" }
```

If the client supports MCP elicitation, the server asks which runner to use. Pass `"elicit": false` to skip.

#### Persistent preferences

**You say:** "Use YOLO mode with Gemini Flash from now on."

```json
{ "execution_mode": "yolo", "model": "gemini-3-flash-preview", "max_retries": 5 }
```

Subsequent calls inherit these settings. Preferences persist across MCP sessions until explicitly cleared.

Fallback chain: **explicit parameter → saved preference → per-runner env → global env → hardcoded default**.

</details>

## MCP Tools

All prompt tools run as background tasks — they return a task ID immediately so the client can
poll for results, preventing MCP timeouts for long operations (e.g. YOLO mode: 2–5 minutes).

| Tool | Task? | Description |
|------|-------|-------------|
| `batch_prompt` | Yes | Fan out prompts to multiple runners in parallel; returns `MultiPromptResponse` |
| `prompt` | Yes | Single-runner convenience wrapper; routes to `batch_prompt` |
| `set_preferences` | No | Set or selectively clear persistent defaults for execution mode, model, retries, timeouts, elicitation, and trigger suppression |
| `get_preferences` | No | Retrieve current preferences |
| `clear_preferences` | No | Reset all preferences |
| `set_model_tiers` | No | Save model tier classifications (client sends sampling/benchmark results; server persists) |
| `get_model_tiers` | No | Retrieve saved model tier classifications |

<details>
<summary><h3>Tool API Reference</h3></summary>

#### `batch_prompt`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `tasks` | Yes | — | List of task objects (see below) |
| `max_concurrency` | No | `3` | Max parallel agent invocations |
| `elicit` | No | pref or `true` | Enable/disable interactive elicitation for this call |

**Task object fields:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `cli` | No | — | Runner name (e.g. `"gemini"`); if omitted, elicitation asks which runner to use |
| `prompt` | Yes | — | Prompt text |
| `label` | No | auto | Display label for results |
| `context` | No | `{}` | Optional context metadata dict |
| `execution_mode` | No | pref or `"default"` | `"default"` or `"yolo"` |
| `model` | No | pref or CLI default | Model name override |
| `max_retries` | No | pref or env default | Max retry attempts for transient errors |
| `output_limit` | No | pref or env default | Max output bytes |
| `timeout` | No | pref or env default | Subprocess timeout in seconds |
| `retry_base_delay` | No | pref or env default | Base delay for exponential backoff |
| `retry_max_delay` | No | pref or env default | Max delay cap for backoff |

> **Note:** `elicit` is a batch-level parameter. When enabled, the server runs a single upfront elicitation pass across all tasks rather than prompting per-task.

#### `prompt`

Same parameters as a single task object in `batch_prompt`, plus `elicit` (batch-level in `batch_prompt`, per-call here).

#### `set_preferences`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `execution_mode` | No | — | `"default"` or `"yolo"` |
| `model` | No | — | Model name (e.g. `"gemini-3-flash-preview"`) |
| `max_retries` | No | — | Max total attempts (≥1; 1 = no retries) |
| `output_limit` | No | — | Max output bytes (≥1) |
| `timeout` | No | — | Subprocess timeout seconds (≥1) |
| `retry_base_delay` | No | — | Backoff base delay seconds (≥0) |
| `retry_max_delay` | No | — | Backoff max delay seconds (≥0) |
| `elicit` | No | `true` | Enable/disable elicitation |
| `confirm_yolo` | No | `true` | Prompt before YOLO mode (auto-suppressed after first accept) |
| `confirm_vague_prompt` | No | `true` | Prompt on very short prompts |
| `confirm_high_retries` | No | `true` | Prompt when max_retries > 5 |
| `confirm_large_batch` | No | `true` | Prompt when batch > 5 tasks |
| `clear_*` | No | `false` | Clear any field individually (e.g. `clear_model: true`) |

#### `get_preferences` / `clear_preferences`

`get_preferences` — no parameters, returns all fields (`null` when unset).
`clear_preferences` — no parameters, resets all to `null`. Does **not** clear model tiers.

#### `set_model_tiers`

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `tiers` | Yes | — | Dict mapping model names to tiers (`"quick"`, `"standard"`, `"thorough"`) |

Persists tier classifications. Clients typically call once via sampling or benchmark fetch.

#### `get_model_tiers`

No parameters. Returns saved tiers as `dict[str, str]`, or `{}` if none saved.

</details>

### Managing Preferences

| Operation | Tool | Notes |
|-----------|------|-------|
| Set fields | `set_preferences` | Persists across sessions |
| Read values | `get_preferences` | `null` for unset fields |
| Clear all | `clear_preferences` | Does not clear model tiers |
| Clear one field | `set_preferences` with `clear_*: true` | Others preserved |
| Suppress elicitation | `set_preferences` with `confirm_*: false` | YOLO/batch/retry auto-suppress after accept |
| Re-enable prompt | `set_preferences` with `clear_confirm_*: true` | Resets to default |
| Save/read tiers | `set_model_tiers` / `get_model_tiers` | Persists across sessions |

## MCP Prompts

Nexus MCP provides 10 discoverable prompt templates that clients can browse via `list_prompts()` and render via `get_prompt(name, args)`. Each prompt returns structured messages with expert framing — the client decides how (or whether) to use them.

**Design principle:** Server informs, client decides. Prompts provide the scaffold (role, structure, methodology); the client decides runner, model, depth, and orchestration. Prompts are completely optional — existing `prompt`/`batch_prompt` tools work exactly as before.

| Prompt | Tags | Parameters | Purpose |
|--------|------|------------|---------|
| `code_review` | analysis | `file`, `instructions` | Structured code review with findings by severity |
| `debug` | analysis | `error`, `context`, `file` | Systematic diagnosis: reproduce, isolate, root cause, fix |
| `quick_triage` | analysis | `description`, `file` | Fast assessment: what's wrong, severity, next step |
| `research` | analysis | `topic`, `scope` | Structured research with source citations |
| `second_opinion` | analysis | `original_output`, `question` | Independent review of another AI's output |
| `implement_feature` | generation | `description`, `language`, `constraints` | Feature implementation with quality checklist |
| `refactor` | generation | `file`, `goal`, `constraints` | Behavior-preserving restructuring |
| `bulk_generate` | generation | `template`, `variables` | Expand template across variable sets |
| `write_tests` | testing | `file`, `framework`, `coverage_goal` | Test generation with configurable coverage approach |
| `compare_models` | comparison | `prompt`, `criteria` | Multi-runner comparison framework |

<details>
<summary><strong>Example — using a prompt template</strong></summary>

```
# 1. Client discovers available prompts
list_prompts() → sees "code_review", "debug", "compare_models", etc.

# 2. Client renders a prompt with arguments
get_prompt("code_review", {file: "src/auth.py", instructions: "security vulnerabilities"})

# 3. Server returns structured messages
→ PromptResult(
    messages=[
      Message("You are a senior code reviewer...", role="assistant"),
      Message("Review the file `src/auth.py`...\nFocus: security vulnerabilities\n...", role="user"),
    ],
    description="Code review of src/auth.py"
  )

# 4. Client feeds messages into prompt/batch_prompt with chosen runner+model
prompt(cli="claude", prompt=<rendered messages>)
```

</details>

## MCP Resources

Read-only data endpoints that clients query for runner metadata, configuration, and preferences.

| Resource URI | Description |
|---|---|
| `nexus://runners` | All registered CLI runners with models (enriched with tier data), modes, availability |
| `nexus://runners/{cli}` | Single runner details by name (URI template) |
| `nexus://config` | Resolved operational config defaults (timeouts, retries, output limits) |
| `nexus://preferences` | Current preferences with config fallback |

Models in `nexus://runners` include tier data: `{"name": "gemini-2.5-flash", "tier": "quick"}`. Tiers are `quick` (fast/cheap), `standard` (balanced), or `thorough` (max quality). Models with only heuristic tiers appear in `unclassified_models` — calling `set_model_tiers` moves them out.

<details>
<summary><strong>Model tier enrichment examples</strong></summary>

**Before `set_model_tiers`** — all tiers are heuristic guesses, all models are unclassified:

```json
{
  "models": [
    {"name": "gemini-3.1-pro-preview", "tier": "thorough"},
    {"name": "gemini-2.5-flash", "tier": "quick"},
    {"name": "gemini-2.5-flash-lite", "tier": "quick"}
  ],
  "unclassified_models": ["gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
}
```

**After `set_model_tiers`** — saved tiers replace heuristics, classified models leave the list:

```json
{
  "models": [
    {"name": "gemini-3.1-pro-preview", "tier": "thorough"},
    {"name": "gemini-2.5-flash", "tier": "quick"},
    {"name": "gemini-2.5-flash-lite", "tier": "quick"}
  ],
  "unclassified_models": []
}
```

</details>

<details>
<summary><h2>Configuration</h2></summary>

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
| `NEXUS_{AGENT}_MODEL` | `NEXUS_GEMINI_MODEL=gemini-3-flash-preview` | Default model for this runner |
| `NEXUS_{AGENT}_MODELS` | `NEXUS_GEMINI_MODELS=gemini-3-flash-preview,gemini-2.5-pro` | Comma-separated model list (surfaced in server instructions) |
| `NEXUS_{AGENT}_TIMEOUT` | `NEXUS_GEMINI_TIMEOUT=900` | Subprocess timeout override |
| `NEXUS_{AGENT}_OUTPUT_LIMIT` | `NEXUS_CODEX_OUTPUT_LIMIT=100000` | Output limit override |
| `NEXUS_{AGENT}_MAX_RETRIES` | `NEXUS_CLAUDE_MAX_RETRIES=5` | Max retry attempts override |
| `NEXUS_{AGENT}_RETRY_BASE_DELAY` | `NEXUS_GEMINI_RETRY_BASE_DELAY=1.0` | Backoff base delay override |
| `NEXUS_{AGENT}_RETRY_MAX_DELAY` | `NEXUS_GEMINI_RETRY_MAX_DELAY=30.0` | Backoff max delay override |
| `NEXUS_{AGENT}_EXECUTION_MODE` | `NEXUS_GEMINI_EXECUTION_MODE=yolo` | Execution mode override |

Invalid per-runner values are silently ignored (the global or hardcoded default is used instead).

</details>

<details>
<summary><h2>Development</h2></summary>

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
│   ├── server.py           # FastMCP server + tools + prompt registration
│   ├── types.py            # Pydantic models
│   ├── exceptions.py       # Exception hierarchy
│   ├── config.py           # Environment variable config
│   ├── store.py            # Persistent backing store access (preferences + tiers)
│   ├── tiers.py            # Heuristic model tier classification
│   ├── elicitation.py      # ElicitationGuard — interactive parameter resolution
│   ├── resources.py        # MCP resources (runners, config, preferences)
│   ├── process.py          # Subprocess wrapper
│   ├── parser.py           # JSON→text fallback parsing
│   ├── cli_detector.py     # CLI binary detection + version checks
│   ├── prompts/
│   │   ├── __init__.py     # register_prompts(mcp) entry point
│   │   ├── analysis.py     # code_review, debug, quick_triage, research, second_opinion
│   │   ├── generation.py   # implement_feature, refactor, bulk_generate
│   │   ├── testing.py      # write_tests
│   │   └── comparison.py   # compare_models
│   └── runners/
│       ├── base.py         # Protocol + ABC
│       ├── factory.py      # RunnerFactory
│       ├── claude.py       # ClaudeRunner
│       ├── codex.py        # CodexRunner
│       ├── gemini.py       # GeminiRunner
│       └── opencode.py     # OpenCodeRunner
├── tests/
│   ├── unit/               # Fast, mocked tests
│   │   └── prompts/        # Prompt template tests
│   ├── e2e/                # End-to-end MCP protocol tests
│   ├── integration/        # Real CLI tests
│   └── fixtures.py         # Shared test utilities
├── .github/
│   └── workflows/          # CI, security, dependabot
├── pyproject.toml          # Dependencies + tool config
└── .pre-commit-config.yaml # Git hooks configuration
```

</details>

## License

MIT
