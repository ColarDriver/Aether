from __future__ import annotations

import ast
from pathlib import Path

from aether.gateway.handlers.service_errors import service_error_to_gateway
from aether.gateway.protocol import ERROR_APPLICATION, ERROR_INVALID_PARAMS
from aether.services.common import ServiceNotFoundError, ServiceValidationError


MIGRATED_HANDLERS = (
    "prefs_methods.py",
    "session_methods.py",
    "tools_methods.py",
    "providers_methods.py",
)

FORBIDDEN_LOW_LEVEL_IMPORTS = (
    "aether.cli.prefs",
    "aether.cli.sessions",
    "aether.tools.builtins",
    "aether.config.auxiliary_slots",
    "aether.config.provider_runtime",
    "aether.runtime.credentials",
)


def test_service_errors_map_to_existing_gateway_error_codes() -> None:
    invalid = service_error_to_gateway(ServiceValidationError("bad"))
    missing = service_error_to_gateway(ServiceNotFoundError("missing", details={"id": "x"}))

    assert invalid.code == ERROR_INVALID_PARAMS
    assert missing.code == ERROR_APPLICATION
    assert missing.data == {"id": "x"}


def test_migrated_gateway_handlers_do_not_import_low_level_service_dependencies() -> None:
    root = Path(__file__).resolve().parents[2] / "gateway" / "handlers"
    offenders: list[str] = []
    for name in MIGRATED_HANDLERS:
        path = root / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            for module in imported:
                if module in FORBIDDEN_LOW_LEVEL_IMPORTS:
                    offenders.append(f"{name} -> {module}")

    assert offenders == []
