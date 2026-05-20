"""Gateway-independent service error types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ServiceError(Exception):
    """Base class for public-safe application service failures."""

    message: str
    code: str = "service_error"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


class ServiceValidationError(ServiceError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, code="validation_error", details=details or {})


class ServiceNotFoundError(ServiceError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, code="not_found", details=details or {})


class ServiceConflictError(ServiceError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, code="conflict", details=details or {})


class ServiceUnavailableError(ServiceError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, code="unavailable", details=details or {})


class ServiceExecutionError(ServiceError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, code="execution_error", details=details or {})


__all__ = [
    "ServiceConflictError",
    "ServiceError",
    "ServiceExecutionError",
    "ServiceNotFoundError",
    "ServiceUnavailableError",
    "ServiceValidationError",
]
