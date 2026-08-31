"""Static architecture boundary checks for production imports."""

import ast
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / "src" / "nexus_mcp"
MCP_PACKAGE_PARTS = ("src", "nexus_mcp", "mcp")


def production_python_files() -> list[Path]:
    """Return every production Python source file in a stable order."""
    return sorted(PRODUCTION_ROOT.rglob("*.py"))


def direct_imports_of(module_name: str, files: Iterable[Path]) -> list[str]:
    """Return direct imports of a module from outside the MCP adapter package."""
    violations: list[str] = []
    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)
        if relative_path.parts[: len(MCP_PACKAGE_PARTS)] == MCP_PACKAGE_PARTS:
            continue

        tree = ast.parse(path.read_text(), filename=str(relative_path))
        for node in ast.walk(tree):
            imported_modules: list[str]
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules = [node.module]
            else:
                continue

            if any(
                imported == module_name or imported.startswith(f"{module_name}.")
                for imported in imported_modules
            ):
                violations.append(f"{relative_path}:{node.lineno}")

    return violations


def test_fastmcp_imports_are_confined_to_mcp_package() -> None:
    """FastMCP belongs only to the MCP transport adapter package."""
    violations = direct_imports_of("fastmcp", production_python_files())
    assert violations == [], "Direct FastMCP imports outside nexus_mcp/mcp:\n" + "\n".join(
        violations
    )
