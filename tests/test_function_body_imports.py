"""Imports that sit inside function bodies are only executed on the code path that needs them
(often a GPU-only branch), so a wrong module path would pass every CPU test and fail hours into
a real run. This test walks the syntax tree of the package and checks that every
``from hl.x import y`` / ``import hl.x`` anywhere in the package names a module that exists and,
for from-imports, a name that module defines at top level."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "hl"
FILES = sorted(PKG.rglob("*.py"))


def _module_path(mod: str) -> Path | None:
    rel = Path(*mod.split("."))
    for cand in (ROOT / rel.with_suffix(".py"), ROOT / rel / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree) if False else tree.body:  # top level only
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.If, ast.Try, ast.With)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
                elif isinstance(sub, ast.ImportFrom):
                    names.update(a.asname or a.name for a in sub.names)
    return names


@pytest.mark.parametrize("path", FILES, ids=[str(p.relative_to(ROOT)) for p in FILES])
def test_imports_of_this_package_resolve(path):
    tree = ast.parse(path.read_text())
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (
            node.module == "hl" or node.module.startswith("hl.")
        ):
            target = _module_path(node.module)
            if target is None:
                problems.append(f"line {node.lineno}: no module {node.module}")
                continue
            defined = _top_level_names(target)
            for a in node.names:
                if a.name == "*":
                    continue
                if a.name in defined:
                    continue
                if _module_path(f"{node.module}.{a.name}") is not None:
                    continue  # importing a submodule from a package
                problems.append(f"line {node.lineno}: {node.module} does not define {a.name}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if (a.name == "hl" or a.name.startswith("hl.")) and _module_path(a.name) is None:
                    problems.append(f"line {node.lineno}: no module {a.name}")
    assert not problems, f"{path.relative_to(ROOT)}: " + "; ".join(problems)


def test_no_sys_path_edits_in_package():
    offenders = [str(p.relative_to(ROOT)) for p in FILES if "sys.path.insert" in p.read_text() or "sys.path.append" in p.read_text()]
    assert not offenders, f"sys.path edits are not allowed in the package: {offenders}"
