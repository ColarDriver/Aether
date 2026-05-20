"""Gateway error mapping for service-layer failures."""

from __future__ import annotations

from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    GatewayError,
)
from aether.services.common import (
    ServiceConflictError,
    ServiceError,
    ServiceNotFoundError,
    ServiceValidationError,
)


def service_error_to_gateway(error: ServiceError) -> GatewayError:
    if isinstance(error, ServiceValidationError):
        return GatewayError(
            error.message,
            code=ERROR_INVALID_PARAMS,
            data=error.details or None,
        )
    if isinstance(error, (ServiceNotFoundError, ServiceConflictError)):
        return GatewayError(
            error.message,
            code=ERROR_APPLICATION,
            data=error.details or None,
        )
    return GatewayError(
        error.message,
        code=ERROR_APPLICATION,
        data=error.details or None,
    )


__all__ = ["service_error_to_gateway"]
