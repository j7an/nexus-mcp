# src/nexus_mcp/cli_detector.py
"""CLI detection, version parsing, and capability checking.

Provides:
- detect_cli(): Check if a CLI binary exists in PATH (sync, via shutil.which)
- get_cli_version(): Run '<cli> --version' and parse the output (sync, via subprocess.run)
- parse_version(): Extract semver from CLI version output
- supports_json_output(): Check if CLI version supports JSON output
- get_cli_capabilities(): Aggregate capabilities for a CLI + version
"""

import re
import shutil
import subprocess
from dataclasses import dataclass

from packaging import version as pkg_version


@dataclass
class CLIInfo:
    """CLI detection result."""

    found: bool
    path: str | None = None
    version: str | None = None


def detect_cli(cli_name: str) -> CLIInfo:
    """Detect if CLI binary exists in PATH."""
    path = shutil.which(cli_name)
    if path:
        return CLIInfo(found=True, path=path)
    return CLIInfo(found=False)


def get_cli_version(cli_name: str) -> str | None:
    """Run '<cli> --version' and parse the version string.

    Sync — uses subprocess.run(). Acceptable at runner construction time (<1s).
    """
    try:
        result = subprocess.run(
            [cli_name, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return parse_version(result.stdout, cli=cli_name)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def parse_version(version_output: str, cli: str) -> str | None:
    """Parse version string from CLI --version output."""
    patterns: dict[str, str] = {
        "gemini": r"v?(\d+\.\d+\.\d+(?:-[\w.]+)?)",
        "codex": r"version\s+(\d+\.\d+\.\d+)",
        "claude": r"v?(\d+\.\d+\.\d+)",
    }
    pattern = patterns.get(cli)
    if not pattern:
        return None
    match = re.search(pattern, version_output)
    return match.group(1) if match else None


def supports_json_output(cli: str, version: str) -> bool:
    """Check if CLI version supports JSON output."""
    if cli == "gemini":
        try:
            # Strip pre-release suffix: "0.6.0-preview.4" → "0.6.0"
            base_version = version.split("-")[0]
            v = pkg_version.parse(base_version)
            required = pkg_version.parse("0.6.0")
            return v >= required
        except pkg_version.InvalidVersion:
            return False
    elif cli in ("codex", "claude"):
        return True
    return False


@dataclass
class CLICapabilities:
    """Feature capabilities for a specific CLI + version."""

    found: bool
    supports_json: bool = False
    supports_model_flag: bool = False
    fallback_to_text: bool = False


def get_cli_capabilities(cli: str, version: str | None) -> CLICapabilities:
    """Determine CLI capabilities based on version."""
    if version is None:
        return CLICapabilities(found=False)

    json_supported = supports_json_output(cli, version)
    return CLICapabilities(
        found=True,
        supports_json=json_supported,
        supports_model_flag=True,
        fallback_to_text=not json_supported,
    )
