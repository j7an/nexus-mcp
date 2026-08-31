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

        tree = ast.parse(path.read_text(), filename=str(relative_path))
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
        tree = ast.parse(path.read_text(), filename=str(relative_path))
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


def _static_export_strings(
    node: ast.AST,
    bindings: dict[str, tuple[str, ...] | None],
) -> tuple[str, ...] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_export_strings(node.left, bindings)
        right = _static_export_strings(node.right, bindings)
        return None if left is None or right is None else left + right
    if not isinstance(node, ast.List | ast.Tuple):
        return None

    exports: list[str] = []
    for element in node.elts:
        value = element.value if isinstance(element, ast.Starred) else element
        resolved = _static_export_strings(value, bindings)
        if resolved is None:
            return None
        exports.extend(resolved)
    return tuple(exports)


def _target_names(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)
    elif isinstance(target, ast.List | ast.Tuple):
        for element in target.elts:
            yield from _target_names(element)


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
            if alias.name != "*":
                yield alias.asname or alias.name
    elif isinstance(statement, ast.For | ast.AsyncFor):
        yield from _target_names(statement.target)
    elif isinstance(statement, ast.With | ast.AsyncWith):
        for item in statement.items:
            if item.optional_vars is not None:
                yield from _target_names(item.optional_vars)


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


def _is_all_alias(name: str, object_roots: dict[str, str], all_root: str | None) -> bool:
    return name == "__all__" or (all_root is not None and object_roots.get(name) == all_root)


def _target_references_all_alias(
    target: ast.AST,
    object_roots: dict[str, str],
    all_root: str | None,
    *,
    direct_alias: bool,
) -> bool:
    if isinstance(target, ast.Name):
        return target.id == "__all__" or (
            direct_alias and _is_all_alias(target.id, object_roots, all_root)
        )
    return any(
        isinstance(candidate, ast.Name) and _is_all_alias(candidate.id, object_roots, all_root)
        for candidate in ast.walk(target)
    )


def _target_references_all(target: ast.AST) -> bool:
    return _target_references_all_alias(target, {}, None, direct_alias=False)


def _mutates_all_alias(
    statement: ast.stmt,
    object_roots: dict[str, str],
    all_root: str | None,
) -> bool:
    if isinstance(statement, ast.Assign):
        return any(
            _target_references_all_alias(target, object_roots, all_root, direct_alias=False)
            for target in statement.targets
        )
    if isinstance(statement, ast.AnnAssign | ast.Delete):
        targets = statement.targets if isinstance(statement, ast.Delete) else (statement.target,)
        return any(
            _target_references_all_alias(target, object_roots, all_root, direct_alias=False)
            for target in targets
        )
    if isinstance(statement, ast.AugAssign):
        return _target_references_all_alias(
            statement.target, object_roots, all_root, direct_alias=True
        )
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and isinstance(statement.value.func.value, ast.Name)
        and _is_all_alias(statement.value.func.value.id, object_roots, all_root)
    )


def _explicit_public_exports(
    tree: ast.Module,
) -> tuple[tuple[str, ...], list[tuple[int, str]]]:
    bindings: dict[str, tuple[str, ...] | None] = {}
    object_roots: dict[str, str] = {}
    all_root: str | None = None
    exports: tuple[str, ...] | None = ()
    dynamic_updates: list[tuple[int, str]] = []

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            resolved = _static_export_strings(statement.value, bindings)
            value_root = (
                object_roots.get(statement.value.id, statement.value.id)
                if isinstance(statement.value, ast.Name)
                else f"assignment:{statement.lineno}:{statement.col_offset}"
            )
            all_targets = [
                target
                for target in statement.targets
                if isinstance(target, ast.Name) and target.id == "__all__"
            ]
            if all_targets:
                all_root = value_root
                exports = resolved
                if resolved is None:
                    dynamic_updates.append((statement.lineno, "non-analyzable __all__ assignment"))
                bindings["__all__"] = resolved
            elif any(
                _target_references_all_alias(target, object_roots, all_root, direct_alias=False)
                for target in statement.targets
            ):
                dynamic_updates.append((statement.lineno, "non-analyzable __all__ target update"))

            for target in statement.targets:
                if isinstance(target, ast.Name):
                    object_roots[target.id] = value_root
                    if target.id != "__all__":
                        bindings[target.id] = resolved
                elif not isinstance(target, ast.Name):
                    for name in _target_names(target):
                        bindings[name] = None
                        object_roots[name] = f"binding:{statement.lineno}:{name}"
        elif isinstance(statement, ast.AnnAssign):
            resolved = (
                None
                if statement.value is None
                else _static_export_strings(statement.value, bindings)
            )
            value_root = (
                object_roots.get(statement.value.id, statement.value.id)
                if isinstance(statement.value, ast.Name)
                else f"assignment:{statement.lineno}:{statement.col_offset}"
            )
            if isinstance(statement.target, ast.Name) and statement.target.id == "__all__":
                if statement.value is not None:
                    all_root = value_root
                    exports = resolved
                    if resolved is None:
                        dynamic_updates.append(
                            (statement.lineno, "non-analyzable __all__ assignment")
                        )
                    bindings["__all__"] = resolved
            elif _target_references_all_alias(
                statement.target, object_roots, all_root, direct_alias=False
            ):
                dynamic_updates.append((statement.lineno, "non-analyzable __all__ target update"))
            else:
                for name in _target_names(statement.target):
                    bindings[name] = resolved
                    object_roots[name] = value_root
        elif isinstance(statement, ast.AugAssign) and _target_references_all_alias(
            statement.target, object_roots, all_root, direct_alias=True
        ):
            added = _static_export_strings(statement.value, bindings)
            if (
                not isinstance(statement.target, ast.Name)
                or statement.target.id != "__all__"
                or not isinstance(statement.op, ast.Add)
                or exports is None
                or added is None
            ):
                exports = None
                dynamic_updates.append((statement.lineno, "non-analyzable __all__ update"))
            else:
                exports += added
            bindings["__all__"] = exports
        elif isinstance(statement, ast.Delete) and any(
            _target_references_all_alias(target, object_roots, all_root, direct_alias=False)
            for target in statement.targets
        ):
            exports = None
            bindings["__all__"] = None
            dynamic_updates.append((statement.lineno, "non-analyzable __all__ target update"))
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and (
                statement.value.func.value.id == "__all__"
                or (
                    all_root is not None
                    and object_roots.get(statement.value.func.value.id) == all_root
                )
            )
        ):
            call = statement.value
            method = call.func.attr
            resolved = (
                _static_export_strings(call.args[0], bindings)
                if len(call.args) == 1 and not call.keywords
                else None
            )
            if (
                exports is None
                or resolved is None
                or method not in {"append", "extend"}
                or (method == "append" and len(resolved) != 1)
            ):
                exports = None
                dynamic_updates.append((statement.lineno, "non-analyzable __all__ method update"))
            else:
                exports += resolved
            for name, root in object_roots.items():
                if root == all_root:
                    bindings[name] = exports
            bindings["__all__"] = exports
        else:
            for group in _nested_statement_groups(statement):
                for nested_statement, _ in _module_scope_statements(group, nested=True):
                    if _mutates_all_alias(nested_statement, object_roots, all_root):
                        exports = None
                        bindings["__all__"] = None
                        dynamic_updates.append(
                            (nested_statement.lineno, "conditional __all__ update")
                        )
                    for name in _statement_bound_names(nested_statement):
                        bindings[name] = None
                        object_roots[name] = f"conditional:{nested_statement.lineno}:{name}"

    return (() if exports is None else exports), dynamic_updates


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
        tree = ast.parse(path.read_text(), filename=str(relative_path))
        exports, dynamic_updates = _explicit_public_exports(tree)
        for export in exports:
            if any(marker in export.casefold() for marker in PROVIDER_SPECIFIC_EXPORT_MARKERS):
                violations.add(f"{relative_path}: provider-specific export {export}")
        for line_number, message in dynamic_updates:
            violations.add(f"{relative_path}:{line_number}: {message}")
        for binding, line_number in _provider_specific_public_bindings(tree):
            violations.add(f"{relative_path}:{line_number}: provider-specific binding {binding}")
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
