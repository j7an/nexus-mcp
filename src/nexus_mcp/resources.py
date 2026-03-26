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

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ResourceError

from nexus_mcp.cli_detector import detect_cli, get_cli_version
from nexus_mcp.config import _get_merged_defaults, get_runner_defaults, get_runner_models
from nexus_mcp.preferences import _get_session_preferences
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


async def get_preferences_resource(ctx: Context | None = None) -> str:
    """Return current session preferences with config fallback.

    Resource URI: nexus://preferences

    If ctx is available and session preferences can be read, returns them
    with source='session'. Otherwise falls back to resolved config defaults
    with source='defaults'.
    """
    if ctx is None:
        logger.debug("No context provided, falling back to config defaults")
        defaults = _get_merged_defaults()
        fallback = {
            "execution_mode": defaults.execution_mode,
            "model": defaults.model,
            "max_retries": defaults.max_retries,
            "output_limit": defaults.output_limit,
            "timeout": defaults.timeout,
            "retry_base_delay": defaults.retry_base_delay,
            "retry_max_delay": defaults.retry_max_delay,
            "elicit": None,
            "confirm_yolo": None,
            "confirm_vague_prompt": None,
            "confirm_high_retries": None,
            "confirm_large_batch": None,
        }
        return json.dumps({"source": "defaults", "preferences": fallback})
    try:
        prefs = await _get_session_preferences(ctx)
        return json.dumps({"source": "session", "preferences": prefs.model_dump()})
    except Exception:
        logger.debug("Session preferences unavailable, falling back to config defaults")
        defaults = _get_merged_defaults()
        fallback = {
            "execution_mode": defaults.execution_mode,
            "model": defaults.model,
            "max_retries": defaults.max_retries,
            "output_limit": defaults.output_limit,
            "timeout": defaults.timeout,
            "retry_base_delay": defaults.retry_base_delay,
            "retry_max_delay": defaults.retry_max_delay,
            "elicit": None,
            "confirm_yolo": None,
            "confirm_vague_prompt": None,
            "confirm_high_retries": None,
            "confirm_large_batch": None,
        }
        return json.dumps({"source": "defaults", "preferences": fallback})


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
    mcp.resource(
        "nexus://preferences", mime_type="application/json", annotations=_RESOURCE_ANNOTATIONS
    )(get_preferences_resource)
