"""HTTP error mapping for service-layer failures."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aether.services.common import (
    ServiceConflictError,
    ServiceError,
    ServiceNotFoundError,
    ServiceValidationError,
)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=status_code_for_service_error(exc),
            content={"error": exc.to_public_dict()},
        )


def status_code_for_service_error(exc: ServiceError) -> int:
    if isinstance(exc, ServiceValidationError):
        return 400
    if isinstance(exc, ServiceNotFoundError):
        return 404
    if isinstance(exc, ServiceConflictError):
        return 409
    return 500


__all__ = ["install_error_handlers", "status_code_for_service_error"]
