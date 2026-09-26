"""Root conftest shared by unit, e2e, and integration tests."""

import os

os.environ.setdefault("FASTMCP_MCP_CAMELCASE_COMPAT", "false")
