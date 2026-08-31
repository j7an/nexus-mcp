"""Static architecture boundary checks for production imports and legacy seams."""

import ast
import re
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / "src" / "nexus_mcp"
CORE_ROOT = PRODUCTION_ROOT / "core"
JOBS_ROOT = PRODUCTION_ROOT / "jobs"
BACKENDS_ROOT = PRODUCTION_ROOT / "backends"

MCP_PACKAGE_PARTS = ("src", "nexus_mcp", "mcp")
SQLITE_STORE_PARTS = ("src", "nexus_mcp", "jobs", "sqlite_store.py")
SQLITE_MIGRATIONS_PARTS = ("src", "nexus_mcp", "jobs", "migrations")
LEGACY_COMMAND_ALLOWED_PREFIXES = (
    ("src", "nexus_mcp", "runners"),
    ("src", "nexus_mcp", "legacy"),
    ("tests", "unit", "runners"),
    ("tests", "unit", "legacy"),
    ("tests", "integration"),
)
LEGACY_COMMAND_ALLOWED_FILES = {
    ("tests", "fixtures.py"),
    ("tests", "unit", "test_architecture_boundaries.py"),
}
LEGACY_COMMAND_PATTERNS = (
    (re.compile(r"\bcodex(?:[\W_]+)exec\b", re.IGNORECASE), "codex exec"),
    (re.compile(r"\bopencode(?:[\W_]+)run\b", re.IGNORECASE), "opencode run"),
)
CORE_FORBIDDEN_IMPORT_ROOTS = (
    "runners",
    "process",
    "parser",
    "http_client",
    "nexus_mcp.runners",
    "nexus_mcp.process",
    "nexus_mcp.parser",
    "nexus_mcp.http_client",
)
PROVIDER_SPECIFIC_EXPORT_MARKERS = ("codex", "opencode", "claude")


def python_files_under(*roots: Path) -> list[Path]:
    """Return Python source files below the supplied roots in a stable order."""
    return sorted(path for root in roots for path in root.rglob("*.py"))


def production_python_files() -> list[Path]:
    """Return every production Python source file in a stable order."""
    return python_files_under(PRODUCTION_ROOT)


def project_python_files() -> list[Path]:
    """Return production and test Python source files in a stable order."""
    return python_files_under(PRODUCTION_ROOT, PROJECT_ROOT / "tests")


def _path_starts_with(path: Path, prefix: tuple[str, ...]) -> bool:
    return path.parts[: len(prefix)] == prefix


def _is_allowed_path(
    path: Path,
    *,
    prefixes: Iterable[tuple[str, ...]] = (),
    exact_files: Iterable[tuple[str, ...]] = (),
) -> bool:
    return path.parts in exact_files or any(_path_starts_with(path, prefix) for prefix in prefixes)


def _imported_modules(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()

    imported: list[str] = []
    if node.module is not None:
        imported.append(node.module)
    if node.module in {None, "nexus_mcp"}:
        prefix = "" if node.module is None else f"{node.module}."
        imported.extend(f"{prefix}{alias.name}" for alias in node.names if alias.name != "*")
    return tuple(imported)


def direct_imports_of(
    module_name: str,
    files: Iterable[Path],
    *,
    allowed_prefixes: Iterable[tuple[str, ...]] = (),
    allowed_files: Iterable[tuple[str, ...]] = (),
) -> list[str]:
    """Return direct imports of a module from outside explicitly allowed paths."""
    violations: set[str] = set()
    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)
        if _is_allowed_path(
            relative_path,
            prefixes=allowed_prefixes,
            exact_files=allowed_files,
        ):
            continue

        tree = ast.parse(path.read_text(), filename=str(relative_path))
        for node in ast.walk(tree):
            if any(
                imported == module_name or imported.startswith(f"{module_name}.")
                for imported in _imported_modules(node)
            ):
                violations.add(f"{relative_path}:{node.lineno}")

    return sorted(violations)


def _legacy_command_violations(files: Iterable[Path]) -> list[str]:
    violations: set[str] = set()
    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)
        if _is_allowed_path(
            relative_path,
            prefixes=LEGACY_COMMAND_ALLOWED_PREFIXES,
            exact_files=LEGACY_COMMAND_ALLOWED_FILES,
        ):
            continue
        source = path.read_text()
        for pattern, command in LEGACY_COMMAND_PATTERNS:
            for match in pattern.finditer(source):
                line_number = source.count("\n", 0, match.start()) + 1
                violations.add(f"{relative_path}:{line_number}: {command}")
    return sorted(violations)


def _references_attribute(node: ast.AST, attribute_name: str) -> bool:
    return any(
        isinstance(candidate, ast.Attribute) and candidate.attr == attribute_name
        for candidate in ast.walk(node)
    )


def _is_direct_providers_mutation(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "providers" and isinstance(node.ctx, (ast.Store, ast.Del))
    if isinstance(node, ast.Subscript):
        return isinstance(node.ctx, (ast.Store, ast.Del)) and _references_attribute(
            node.value, "providers"
        )
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and _references_attribute(node.func.value, "providers")
    )


def _forbidden_runtime_internal_violations(files: Iterable[Path]) -> list[str]:
    violations: set[str] = set()
    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)
        tree = ast.parse(path.read_text(), filename=str(relative_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {
                "_state_store",
                "_local_provider",
            }:
                violations.add(f"{relative_path}:{node.lineno}: .{node.attr}")
            elif _is_direct_providers_mutation(node):
                violations.add(f"{relative_path}:{node.lineno}: direct providers mutation")
    return sorted(violations)


def _is_provider_module(module_name: str) -> bool:
    for part in module_name.casefold().split("."):
        if (
            part in {"provider", "providers"}
            or part.startswith(("provider_", "providers_"))
            or part.endswith(("_provider", "_providers"))
        ):
            return True
    return False


def _forbidden_core_import_violations(files: Iterable[Path]) -> list[str]:
    violations: set[str] = set()
    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)
        tree = ast.parse(path.read_text(), filename=str(relative_path))
        for node in ast.walk(tree):
            for imported in _imported_modules(node):
                if _is_provider_module(imported) or any(
                    imported == root or imported.startswith(f"{root}.")
                    for root in CORE_FORBIDDEN_IMPORT_ROOTS
                ):
                    violations.add(f"{relative_path}:{node.lineno}: {imported}")
    return sorted(violations)


def _explicit_public_exports(tree: ast.Module) -> Iterable[tuple[str, int]]:
    for node in tree.body:
        value: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            )
        ) or (
            isinstance(node, (ast.AnnAssign, ast.AugAssign))
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value = node.value
        if value is None:
            continue
        for candidate in ast.walk(value):
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                yield candidate.value, candidate.lineno


def _provider_specific_core_exports(files: Iterable[Path]) -> list[str]:
    violations: set[str] = set()
    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)
        tree = ast.parse(path.read_text(), filename=str(relative_path))
        for export, line_number in _explicit_public_exports(tree):
            if any(marker in export.casefold() for marker in PROVIDER_SPECIFIC_EXPORT_MARKERS):
                violations.add(f"{relative_path}:{line_number}: {export}")
    return sorted(violations)


def test_fastmcp_imports_are_confined_to_mcp_package() -> None:
    """FastMCP belongs only to the MCP transport adapter package."""
    violations = direct_imports_of(
        "fastmcp",
        production_python_files(),
        allowed_prefixes=(MCP_PACKAGE_PARTS,),
    )
    assert violations == [], "Direct FastMCP imports outside nexus_mcp/mcp:\n" + "\n".join(
        violations
    )


def test_sqlite_imports_are_confined_to_store_and_migrations() -> None:
    """SQLite belongs only to its concrete store and schema migration package."""
    violations = direct_imports_of(
        "sqlite3",
        production_python_files(),
        allowed_prefixes=(SQLITE_MIGRATIONS_PARTS,),
        allowed_files=(SQLITE_STORE_PARTS,),
    )
    assert violations == [], "Direct sqlite3 imports outside the SQLite adapter:\n" + "\n".join(
        violations
    )


def test_legacy_command_literals_stay_in_legacy_implementation_and_tests() -> None:
    """Legacy CLI command construction cannot spread into the new architecture."""
    violations = _legacy_command_violations(project_python_files())
    assert violations == [], "Legacy CLI commands outside legacy paths:\n" + "\n".join(violations)


def test_new_core_job_and_backend_code_avoids_fastmcp_runtime_internals() -> None:
    """Framework-independent packages cannot mutate FastMCP runtime internals."""
    files = python_files_under(CORE_ROOT, JOBS_ROOT, BACKENDS_ROOT)
    violations = _forbidden_runtime_internal_violations(files)
    assert violations == [], "FastMCP runtime internals in core/job/backend code:\n" + "\n".join(
        violations
    )


def test_core_imports_are_framework_and_provider_independent() -> None:
    """Core contracts cannot depend on runner, process, parser, HTTP, or provider modules."""
    violations = _forbidden_core_import_violations(python_files_under(CORE_ROOT))
    assert violations == [], "Forbidden imports in nexus_mcp/core:\n" + "\n".join(violations)


def test_core_public_exports_are_provider_neutral() -> None:
    """Core public names cannot encode Codex, OpenCode, or Claude concepts."""
    violations = _provider_specific_core_exports(python_files_under(CORE_ROOT))
    assert violations == [], "Provider-specific public exports in nexus_mcp/core:\n" + "\n".join(
        violations
    )
