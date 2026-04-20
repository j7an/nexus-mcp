# OpenCode Server Setup Guide — experimental

Run an isolated OpenCode server via Docker for HTTP-based agent execution. This enables session management, file search, workspace tools, permissions, and questions — capabilities not available through the CLI subprocess runner.

> ⚠️ **Experimental** — This integration has not been validated end-to-end by the maintainer. Expect rough edges in setup, auth, and tool exposure. The MCP tools surfaced from upstream OpenCode track the upstream project and may change without notice. Feedback and bug reports are welcome.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) (v2+)
- An API key for at least one LLM provider (Anthropic, OpenAI, Google, etc.)

## Quick Start

```bash
# 1. Copy the environment template
cp .env.example .env

# 2. Set your project directory (the directory OpenCode will work in)
#    Edit .env and change PROJECT_DIR to your project path:
#    PROJECT_DIR=/path/to/your/project

# 3. Start the server
docker compose up -d

# 4. Authenticate with your provider
docker exec -it opencode-server opencode auth login

# 5. Verify health
curl -u opencode:nexus http://localhost:4096/global/health
# Expected: {"healthy":true,"version":"..."}
```

## Authentication

Two methods, both using OpenCode's native auth system:

### Interactive (recommended for local development)

```bash
docker exec -it opencode-server opencode auth login
```

This launches OpenCode's built-in auth flow. Credentials are saved to the persistent volume (`opencode-data`) and survive container restarts.

### Environment variables (for CI/headless)

Set provider API keys in `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

OpenCode auto-detects these standard environment variables. See `.env.example` for the full list of supported providers (20+).

## Workspace Configuration

### One container per project

The `PROJECT_DIR` variable in `.env` controls which directory is mounted as `/workspace` inside the container:

```env
PROJECT_DIR=/Users/you/Documents/github/my-app
```

OpenCode operates on files inside `/workspace`. Changing `PROJECT_DIR` requires restarting the container:

```bash
# Edit .env with new PROJECT_DIR, then:
docker compose restart
```

### Multiple projects simultaneously

Run separate containers on different ports:

```bash
# Terminal 1: project A on port 4096
PROJECT_DIR=/path/to/project-a OPENCODE_PORT=4096 docker compose up -d

# Terminal 2: project B on port 4097
PROJECT_DIR=/path/to/project-b OPENCODE_PORT=4097 docker compose up -d
```

Configure nexus-mcp to connect to the appropriate port per project.

## Network Security

### Default: localhost only

The Docker port binding defaults to `127.0.0.1` (localhost only):

```yaml
ports:
  - "127.0.0.1:${OPENCODE_PORT:-4096}:4096"
```

This means **only processes on the same machine** can connect to the OpenCode server. This includes nexus-mcp, curl, and your browser — but not other devices on your network.

### Why localhost binding matters

The OpenCode server exposes powerful capabilities behind HTTP Basic Auth:

- **File read/write** — read and modify any file in your mounted project
- **Shell execution** — run arbitrary commands in the project context
- **Provider credentials** — access stored API keys for all configured providers
- **Session control** — create, abort, and manage coding sessions

The default password (`nexus`) is intentionally simple for local development. This is safe when the server is only reachable from localhost.

### Remote access (VPS / cloud)

If you need to access the server from another machine (e.g., running Docker on a VPS):

> **WARNING:** Changing `127.0.0.1` to `0.0.0.0` exposes the server to your entire network. Anyone who can reach the port gets full access to your project files, shell execution, and provider API keys.

**Required mitigations:**

1. **Change the default password** to a strong, unique value:
   ```env
   OPENCODE_SERVER_PASSWORD=your-strong-random-password-here
   ```

2. **Use a reverse proxy with TLS** — never expose plain HTTP Basic Auth over the network:
   ```
   Internet → Nginx/Traefik (HTTPS) → 127.0.0.1:4096 (HTTP, container)
   ```

3. **Restrict access** via firewall rules or VPN:
   ```bash
   # Example: only allow your IP
   ufw allow from YOUR_IP to any port 4096
   ufw deny 4096
   ```

4. **Prefer SSH tunneling** over direct exposure:
   ```bash
   # From your local machine:
   ssh -L 4096:localhost:4096 user@your-vps
   # Then connect to http://localhost:4096 locally
   ```

## Connecting nexus-mcp

### Environment variables

Set in `.env` (must match the Docker container credentials):

```env
NEXUS_OPENCODE_SERVER_URL=http://localhost:4096
NEXUS_OPENCODE_SERVER_PASSWORD=nexus
NEXUS_OPENCODE_SERVER_USERNAME=opencode
```

### MCP client configuration

Add the OpenCode server env vars to your MCP client config. Example for Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "nexus-mcp": {
      "command": "uvx",
      "args": ["nexus-mcp"],
      "env": {
        "NEXUS_OPENCODE_SERVER_URL": "http://localhost:4096",
        "NEXUS_OPENCODE_SERVER_PASSWORD": "nexus",
        "NEXUS_OPENCODE_SERVER_USERNAME": "opencode"
      }
    }
  }
}
```

When the server is configured and healthy, nexus-mcp automatically registers 38 additional tools and 18 resources for workspace operations, session management, permissions, and questions.

## Health Check

The container includes a built-in health check that polls `GET /global/health` every 30 seconds.

Check health manually:

```bash
# Via curl
curl -u opencode:nexus http://localhost:4096/global/health

# Via Docker
docker inspect --format='{{.State.Health.Status}}' opencode-server
```

The nexus-mcp `nexus://opencode` resource also reports server health status.

## Troubleshooting

### Container won't start

```bash
docker compose logs opencode-server
```

Common causes:
- **Port conflict** — another service on port 4096. Change `OPENCODE_PORT` in `.env`.
- **Image pull failure** — check network connectivity and Docker Hub access.

### Auth fails (401)

The password in `.env` must match between OpenCode server and nexus-mcp:
- `OPENCODE_SERVER_PASSWORD` → used by the Docker container
- `NEXUS_OPENCODE_SERVER_PASSWORD` → used by nexus-mcp to connect

If they don't match, nexus-mcp gets HTTP 401 and logs "OpenCode server returned 401".

### Files not visible to OpenCode

Check `PROJECT_DIR` in `.env`. It must be an absolute path to your project:

```bash
# Wrong (mounts the nexus-mcp directory, not your project)
PROJECT_DIR=.

# Right
PROJECT_DIR=/Users/you/Documents/github/my-app
```

After changing `PROJECT_DIR`, restart: `docker compose restart`

### Health check failing

```bash
# Check if server is starting up (allow 10-30 seconds)
docker compose logs --tail 20 opencode-server

# Check health check status
docker inspect --format='{{json .State.Health}}' opencode-server | python3 -m json.tool
```

If the server logs show it's running but health check fails, the server may need more startup time. The health check has a 10-second `start_period` grace.

## Upgrading

To upgrade OpenCode:

1. Edit `Dockerfile` — change the version tag:
   ```dockerfile
   FROM ghcr.io/anomalyco/opencode:1.4.0
   ```

2. Rebuild and restart:
   ```bash
   docker compose build --no-cache
   docker compose up -d
   ```

Your auth credentials and session data persist in the `opencode-data` volume across upgrades.
