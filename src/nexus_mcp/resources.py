# src/nexus_mcp/resources.py
"""MCP resource definitions for nexus-mcp.

Exposes read-only resources for runner metadata, configuration, and session
preferences. All resources return JSON strings with application/json MIME type.

Resources:
- nexus://runners: All registered CLI runners with full details
- nexus://runners/{cli}: Single runner by name (URI template)
- nexus://config: Resolved operational config defaults
- nexus://preferences: Session preferences with config fallback
"""

import json
import logging

from fastmcp import FastMCP
from fastmcp.exceptions import ResourceError

from nexus_mcp.cli_detector import detect_cli, get_cli_version
from nexus_mcp.config import _get_merged_defaults, get_runner_defaults, get_runner_models
from nexus_mcp.runners.factory import RunnerFactory

logger = logging.getLogger(__name__)

_RESOURCE_ANNOTATIONS = {"readOnlyHint": True, "idempotentHint": True}


def _build_runner_info(cli_name: str) -> dict[str, object]:
    """Build runner metadata dict for a single CLI.

    Aggregates data from RunnerFactory, cli_detector, and config modules.
    """
    cli_info = detect_cli(cli_name)
    defaults = get_runner_defaults(cli_name)
    runner_cls = RunnerFactory.get_runner_class(cli_name)
    models = get_runner_models(cli_name)
    version = get_cli_version(cli_name) if cli_info.found else None

    return {
        "name": cli_name,
        "installed": cli_info.found,
        "path": cli_info.path,
        "version": version,
        "models": list(models),
        "default_model": defaults.model,
        "supported_modes": list(runner_cls._SUPPORTED_MODES),
        "default_timeout": defaults.timeout,
    }


async def get_all_runners() -> str:
    """Return all registered CLI runners with full details.

    Resource URI: nexus://runners
    """
    runners = [_build_runner_info(name) for name in RunnerFactory.list_clis()]
    return json.dumps({"runners": runners})


async def get_runner(cli: str) -> str:
    """Return details for a single CLI runner.

    Resource URI: nexus://runners/{cli}

    Raises:
        ResourceError: If the CLI name is not in RunnerFactory._REGISTRY.
    """
    if cli not in RunnerFactory._REGISTRY:
        raise ResourceError(f"Unknown CLI runner: {cli!r}")
    return json.dumps(_build_runner_info(cli))


async def get_config() -> str:
    """Return resolved operational config defaults.

    Resource URI: nexus://config

    Returns the fully merged defaults (hardcoded + env var overrides).
    Excludes execution_mode and model (exposed via nexus://preferences).
    """
    defaults = _get_merged_defaults()
    return json.dumps(
        {
            "timeout": defaults.timeout,
            "output_limit": defaults.output_limit,
            "max_retries": defaults.max_retries,
            "retry_base_delay": defaults.retry_base_delay,
            "retry_max_delay": defaults.retry_max_delay,
            "tool_timeout": defaults.tool_timeout,
            "cli_detection_timeout": defaults.cli_detection_timeout,
        }
    )


def register_resources(mcp: FastMCP) -> None:
    """Register all MCP resources on the server.

    Called from server.py after tool registration.
    """
    mcp.resource(
        "nexus://runners", mime_type="application/json", annotations=_RESOURCE_ANNOTATIONS
    )(get_all_runners)
    mcp.resource(
        "nexus://runners/{cli}", mime_type="application/json", annotations=_RESOURCE_ANNOTATIONS
    )(get_runner)
    mcp.resource("nexus://config", mime_type="application/json", annotations=_RESOURCE_ANNOTATIONS)(
        get_config
    )
