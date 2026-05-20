from __future__ import annotations

import ast
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2] / "services"

FORBIDDEN_PREFIXES = (
    "aether.gateway.handlers",
    "aether.gateway.protocol",
    "tui",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_services_do_not_import_transport_or_ui_modules() -> None:
    offenders: list[str] = []
    for path in sorted(SERVICE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for imported in _imports(path):
            if imported.startswith(FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(SERVICE_ROOT.parent)} -> {imported}")

    assert offenders == []
