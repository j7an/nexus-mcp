# nexus-mcp
<!-- mcp-name: io.github.j7an/nexus-mcp -->

MCP server that delegates tasks to coding agents — [Claude Code](https://code.claude.com)
via the Claude Agent SDK (and Codex in a later release) — from any MCP client.

## Install

Backends are optional extras; install the ones you use.

| Extra | Backend | Adds |
|---|---|---|
| `claude` | Claude Agent SDK | `claude-agent-sdk` (bundles the Claude Code CLI, ~100 MB) |
| `all` | every backend | |

```bash
uvx --with 'nexus-mcp[all]' nexus-mcp          # run directly
pip install 'nexus-mcp[claude]'                 # or install
```

Claude Code MCP config:

```json
{ "mcpServers": { "nexus": { "command": "uvx", "args": ["--with", "nexus-mcp[all]", "nexus-mcp"] } } }
```

Authentication is the agent's own: log in with `claude`, or set `ANTHROPIC_API_KEY`.

**Windows:** recent `claude-agent-sdk` releases ship no Windows wheel with a bundled CLI;
install Claude Code so `claude` is on `PATH`.

## Tool: `prompt`

Runs one agent turn and returns its final answer.

| Parameter | Default | Meaning |
|---|---|---|
| `backend` | required | `claude` |
| `prompt` | required | Task for the agent |
| `cwd` | required | Absolute project directory |
| `profile` | `read_only` | Permission profile (below) |
| `session_id` | — | Continue this conversation |
| `fork` | `false` | Branch `session_id` into a new conversation |
| `model` | provider default | Model id or alias |
| `timeout` | `600` | Seconds before the turn is cancelled |

Returns `{backend, session_id, output, usage}`. Conversations are stored by the agent itself
(`~/.claude/projects`), so a `session_id` keeps working after nexus-mcp restarts.

### Permission profiles

Actions outside the chosen profile are denied automatically — nobody is prompted.

| Profile | Allows |
|---|---|
| `read_only` | Read/Glob/Grep; `git diff`/`log`/`show` with `--no-ext-diff --no-textconv` |
| `workspace_write` | Plus file edits inside `cwd` and Bash inside the OS sandbox |
| `full_access` | Everything |

## Resource: `nexus://backends`

JSON list of `{name, installed, models, hint}`; `hint` gives the install command for a
missing extra.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `NEXUS_CLAUDE_SETTINGS_PROFILE` | `isolated` | Claude settings to load: `isolated` (none), `project` (project `.claude/`), `inherit` (all, including user settings). Profiles still bound every tool call. |

## Migrating from v1

| v1 | v2 |
|---|---|
| `prompt(cli=…, execution_mode="yolo")` | `prompt(backend=…, profile="full_access", cwd=…)` |
| `batch_prompt` | Parallel `prompt` calls from the client |
| `agent_start` / `agent_status` / `agent_result` | `prompt` (synchronous) |
| `agent_continue` / `agent_fork` | `prompt(session_id=…)` / `prompt(session_id=…, fork=true)` |
| `agent_review` | `prompt(profile="read_only", prompt="Review …")` |
| `agent_cancel` | Cancel the tool call in the client |
| `agent_respond`, elicitation | Removed — choose a profile instead |
| Preferences, model tiers, MCP prompts | Removed |
| OpenCode runners and tools | Removed |
| `NEXUS_*` variables | Removed except `NEXUS_CLAUDE_SETTINGS_PROFILE` |

## Development

```bash
uv sync --all-extras --all-groups
uv run pytest                    # unit + e2e
uv run pytest -m integration     # real CLI (slow, makes model calls)
uv run mypy src/nexus_mcp && uv run ruff check . && uv run ruff format --check .
```

## License

MIT
