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
LEGACY_COMMAND_WORDS = (
    (("codex", "exec"), "codex exec"),
    (("opencode", "run"), "opencode run"),
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
    prefix = "" if node.module is None else f"{node.module}."
    if node.module is not None:
        imported.append(node.module)
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

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        for node in ast.walk(tree):
            if any(
                imported == module_name or imported.startswith(f"{module_name}.")
                for imported in _imported_modules(node)
            ):
                violations.add(f"{relative_path}:{node.lineno}")

    return sorted(violations)


def _static_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        return None if left is None or right is None else left + right
    if not isinstance(node, ast.JoinedStr):
        return None

    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif (
            isinstance(value, ast.FormattedValue)
            and value.conversion == -1
            and value.format_spec is None
            and (formatted := _static_string(value.value)) is not None
        ):
            parts.append(formatted)
        else:
            return None
    return "".join(parts)


def _contiguous_static_strings(nodes: Iterable[ast.AST]) -> Iterable[tuple[str, int]]:
    fragments: list[str] = []
    first_line = 0
    for node in nodes:
        value = _static_string(node)
        if value is None:
            if fragments:
                yield " ".join(fragments), first_line
                fragments = []
            continue
        if not fragments:
            first_line = node.lineno
        fragments.append(value)
    if fragments:
        yield " ".join(fragments), first_line


def _decoded_string_candidates(tree: ast.Module) -> Iterable[tuple[str, int]]:
    candidates: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if (value := _static_string(node)) is not None:
            candidates.add((value, node.lineno))
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            candidates.update(_contiguous_static_strings(node.elts))
        elif isinstance(node, ast.Call):
            candidates.update(_contiguous_static_strings(node.args))
    return sorted(candidates, key=lambda candidate: (candidate[1], candidate[0]))


def _legacy_command(value: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", value.casefold())
    for expected_words, command in LEGACY_COMMAND_WORDS:
        width = len(expected_words)
        if any(
            tuple(words[index : index + width]) == expected_words for index in range(len(words))
        ):
            return command
    return None


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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        for value, line_number in _decoded_string_candidates(tree):
            if (command := _legacy_command(value)) is not None:
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        for node in ast.walk(tree):
            for imported in _imported_modules(node):
                if _is_provider_module(imported) or any(
                    imported == root or imported.startswith(f"{root}.")
                    for root in CORE_FORBIDDEN_IMPORT_ROOTS
                ):
                    violations.add(f"{relative_path}:{node.lineno}: {imported}")
    return sorted(violations)


def _target_names(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)
    elif isinstance(target, ast.List | ast.Tuple):
        for element in target.elts:
            yield from _target_names(element)


def _pattern_bound_names(pattern: ast.pattern) -> Iterable[str]:
    if isinstance(pattern, ast.MatchAs):
        if pattern.name is not None:
            yield pattern.name
        if pattern.pattern is not None:
            yield from _pattern_bound_names(pattern.pattern)
    elif isinstance(pattern, ast.MatchStar):
        if pattern.name is not None:
            yield pattern.name
    elif isinstance(pattern, ast.MatchMapping):
        if pattern.rest is not None:
            yield pattern.rest
        for nested_pattern in pattern.patterns:
            yield from _pattern_bound_names(nested_pattern)
    elif isinstance(pattern, ast.MatchClass):
        for nested_pattern in (*pattern.patterns, *pattern.kwd_patterns):
            yield from _pattern_bound_names(nested_pattern)
    elif isinstance(pattern, ast.MatchSequence | ast.MatchOr):
        for nested_pattern in pattern.patterns:
            yield from _pattern_bound_names(nested_pattern)


def _statement_bound_names(statement: ast.stmt) -> Iterable[str]:
    if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        yield statement.name
    elif isinstance(statement, ast.Assign):
        for target in statement.targets:
            yield from _target_names(target)
    elif isinstance(statement, ast.AnnAssign | ast.AugAssign):
        yield from _target_names(statement.target)
    elif isinstance(statement, ast.Import):
        for alias in statement.names:
            yield alias.asname or alias.name.split(".", maxsplit=1)[0]
    elif isinstance(statement, ast.ImportFrom):
        for alias in statement.names:
            yield "__all__" if alias.name == "*" else alias.asname or alias.name
    elif isinstance(statement, ast.For | ast.AsyncFor):
        yield from _target_names(statement.target)
    elif isinstance(statement, ast.With | ast.AsyncWith):
        for item in statement.items:
            if item.optional_vars is not None:
                yield from _target_names(item.optional_vars)
    elif isinstance(statement, ast.Try | ast.TryStar):
        for handler in statement.handlers:
            if handler.name is not None:
                yield handler.name
    elif isinstance(statement, ast.Match):
        for case in statement.cases:
            yield from _pattern_bound_names(case.pattern)
    elif isinstance(statement, ast.TypeAlias):
        yield from _target_names(statement.name)


def _nested_statement_groups(statement: ast.stmt) -> Iterable[list[ast.stmt]]:
    if isinstance(statement, ast.If | ast.For | ast.AsyncFor | ast.While):
        yield statement.body
        yield statement.orelse
    elif isinstance(statement, ast.With | ast.AsyncWith):
        yield statement.body
    elif isinstance(statement, ast.Try | ast.TryStar):
        yield statement.body
        yield statement.orelse
        yield statement.finalbody
        for handler in statement.handlers:
            yield handler.body
    elif isinstance(statement, ast.Match):
        for case in statement.cases:
            yield case.body


def _module_scope_statements(
    statements: Iterable[ast.stmt],
    *,
    nested: bool = False,
) -> Iterable[tuple[ast.stmt, bool]]:
    for statement in statements:
        yield statement, nested
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for group in _nested_statement_groups(statement):
            yield from _module_scope_statements(group, nested=True)


def _explicit_public_exports(
    tree: ast.Module,
) -> tuple[tuple[str, ...], list[tuple[int, str]]]:
    names = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "__all__"]
    bound_statements = [
        statement
        for statement, _ in _module_scope_statements(tree.body)
        if "__all__" in _statement_bound_names(statement)
    ]
    imported_all = next(
        (
            statement
            for statement in bound_statements
            if isinstance(statement, ast.Import | ast.ImportFrom)
        ),
        None,
    )
    if imported_all is not None:
        return (), [(imported_all.lineno, "nonliteral __all__ use")]
    if not names and not bound_statements:
        return (), []
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    ]
    if len(assignments) != 1:
        anchor = bound_statements[0] if bound_statements else names[0]
        return (), [(anchor.lineno, "nonliteral __all__ use")]
    assignment = assignments[0]
    target = assignment.targets[0]
    value = assignment.value
    if (
        len(assignment.targets) != 1
        or not isinstance(target, ast.Name)
        or target.id != "__all__"
        or not isinstance(value, ast.List | ast.Tuple)
        or not all(
            isinstance(item, ast.Constant) and isinstance(item.value, str) for item in value.elts
        )
        or len(names) != 1
        or names[0] is not target
        or len(bound_statements) != 1
        or bound_statements[0] is not assignment
    ):
        return (), [(assignment.lineno, "nonliteral __all__ use")]
    return tuple(item.value for item in value.elts), []


def _provider_specific_public_bindings(tree: ast.Module) -> Iterable[tuple[str, int]]:
    for statement, _ in _module_scope_statements(tree.body):
        for name in _statement_bound_names(statement):
            if not name.startswith("_") and any(
                marker in name.casefold() for marker in PROVIDER_SPECIFIC_EXPORT_MARKERS
            ):
                yield name, statement.lineno


def _provider_specific_core_exports(files: Iterable[Path]) -> list[str]:
    violations: set[str] = set()
    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(relative_path))
        exports, dynamic_updates = _explicit_public_exports(tree)
        for export in exports:
            if any(marker in export.casefold() for marker in PROVIDER_SPECIFIC_EXPORT_MARKERS):
                violations.add(f"{relative_path}: provider-specific export {export}")
        for line_number, message in dynamic_updates:
            violations.add(f"{relative_path}:{line_number}: {message}")
        for binding, line_number in _provider_specific_public_bindings(tree):
            violations.add(f"{relative_path}:{line_number}: provider-specific binding {binding}")
    return sorted(violations)


def test_dynamic_core_exports_fail_closed() -> None:
    sources = (
        "__all__ = build_exports()",
        "from .exports import names as __all__",
        "from .exports import *",
        '__all__ = ["Safe"]; alias = __all__',
        '__all__ = ["Safe"]; __all__.append("CodexThing")',
        'match ["CodexThing"]:\n    case [*__all__]:\n        pass',
        "def __all__():\n    pass",
        "class __all__:\n    pass",
        "type __all__ = str",
        "try:\n    pass\nexcept Exception as __all__:\n    pass",
    )
    for source in sources:
        _, violations = _explicit_public_exports(ast.parse(source))
        assert violations, source


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


def test_production_code_avoids_fastmcp_runtime_internals() -> None:
    """No production module may touch FastMCP private provider/state internals."""
    violations = _forbidden_runtime_internal_violations(production_python_files())
    assert violations == [], "FastMCP runtime internals in src/:\n" + "\n".join(violations)


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
