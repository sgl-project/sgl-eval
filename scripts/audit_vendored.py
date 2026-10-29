"""Check that static vendored imports stay translated and independent of SE implementation.

Each ``sgl_eval/_vendored/<pkg>/SOURCES.yaml`` names its ``upstream_package``
(``nemo_skills``, ``pier``, ...). Any ``import <upstream_package>...`` left in
that package's files means an ``import_rewrites`` rule is missing.
"""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = ROOT / "sgl_eval" / "_vendored"


def _upstream_packages() -> Dict[Path, str]:
    """Map each vendored package dir to the upstream import root its manifest declares."""
    roots: Dict[Path, str] = {}
    if not VENDOR_ROOT.exists():
        return roots
    for manifest in VENDOR_ROOT.glob("*/SOURCES.yaml"):
        spec = yaml.safe_load(manifest.read_text()) or {}
        upstream = spec.get("upstream_package")
        if not upstream:
            raise ValueError(f"{manifest.relative_to(ROOT)}: missing `upstream_package`")
        roots[manifest.parent] = upstream
    return roots


def _imported_modules(py: Path) -> List[str]:
    relative = py.relative_to(VENDOR_ROOT).with_suffix("")
    package_parts = relative.parts[:-1]
    package = ".".join(("sgl_eval", "_vendored", *package_parts))
    modules = []
    for node in ast.walk(ast.parse(py.read_text(), filename=str(py))):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = resolve_name("." * node.level + module, package)
            modules.append(module)
    return modules


def find_untranslated() -> List[tuple[Path, list[str]]]:
    bad: list[tuple[Path, list[str]]] = []
    for pkg_dir, upstream in _upstream_packages().items():
        for py in pkg_dir.rglob("*.py"):
            hits = [
                module
                for module in _imported_modules(py)
                if module == upstream or module.startswith(upstream + ".")
            ]
            if hits:
                bad.append((py, hits))
    return bad


def find_se_dependencies() -> List[tuple[Path, list[str]]]:
    bad = []
    if not VENDOR_ROOT.exists():
        return bad
    for py in VENDOR_ROOT.rglob("*.py"):
        hits = [
            module
            for module in _imported_modules(py)
            if (module == "sgl_eval" or module.startswith("sgl_eval."))
            and module != "sgl_eval._vendored"
            and not module.startswith("sgl_eval._vendored.")
        ]
        if hits:
            bad.append((py, hits))
    return bad


def main() -> int:
    untranslated = find_untranslated()
    se_dependencies = find_se_dependencies()
    for label, findings in (
        ("Untranslated upstream imports", untranslated),
        ("Vendored imports of SE implementation", se_dependencies),
    ):
        if findings:
            print(f"{label}:")
            for path, hits in findings:
                print(f"  {path.relative_to(ROOT)}: {', '.join(hits)}")
    if untranslated or se_dependencies:
        return 1
    print("OK: static vendored imports are translated and independent of SE implementation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
