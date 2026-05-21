from __future__ import annotations

import importlib

from aether.services.common import (
    ServiceConflictError,
    ServiceError,
    ServiceExecutionError,
    ServiceNotFoundError,
    ServiceUnavailableError,
    ServiceValidationError,
)


SERVICE_PACKAGES = (
    "aether.services.common",
    "aether.services.compact",
    "aether.services.sessions",
    "aether.services.config",
    "aether.services.providers",
    "aether.services.tools",
    "aether.services.skills",
    "aether.services.diagnostics",
    "aether.services.health",
    "aether.services.logs",
    "aether.services.runs",
)


def test_service_packages_import_and_expose_public_exports() -> None:
    for package_name in SERVICE_PACKAGES:
        package = importlib.import_module(package_name)
        assert isinstance(getattr(package, "__all__"), list)


def test_common_service_errors_are_gateway_independent() -> None:
    error = ServiceValidationError("bad input", details={"field": "name"})

    assert isinstance(error, ServiceError)
    assert error.code == "validation_error"
    assert error.to_public_dict() == {
        "code": "validation_error",
        "message": "bad input",
        "details": {"field": "name"},
    }
    assert ServiceNotFoundError("missing").code == "not_found"
    assert ServiceConflictError("conflict").code == "conflict"
    assert ServiceUnavailableError("down").code == "unavailable"
    assert ServiceExecutionError("failed").code == "execution_error"
