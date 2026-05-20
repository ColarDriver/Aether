"""Explicit registry for provider transports."""

from __future__ import annotations

from aether.models.transport.base import ProviderTransport


_REGISTRY: dict[str, type[ProviderTransport]] = {}


def register_transport(api_mode: str, factory: type[ProviderTransport]) -> None:
    """Register a transport factory for an API mode.

    Registering the same factory for the same normalized API mode is
    idempotent. Registering a different factory for an existing API mode is
    rejected so test order and import order cannot silently replace behavior.
    """

    normalized = _normalize_api_mode(api_mode)
    existing = _REGISTRY.get(normalized)
    if existing is factory:
        return
    if existing is not None:
        raise ValueError(
            f"transport already registered for api_mode {normalized!r}: "
            f"{existing.__name__}"
        )
    _REGISTRY[normalized] = factory


def get_transport(api_mode: str) -> ProviderTransport | None:
    """Instantiate the transport for *api_mode*, if registered."""

    factory = _REGISTRY.get(_normalize_api_mode(api_mode))
    return factory() if factory is not None else None


def available_transports() -> tuple[str, ...]:
    """Return registered API modes in stable order."""

    return tuple(sorted(_REGISTRY))


def clear_transports_for_tests() -> None:
    """Clear the registry.

    This is intentionally exported with a test-only name so production callers
    do not treat registry mutation as part of the runtime provider path.
    """

    _REGISTRY.clear()


def _normalize_api_mode(api_mode: str) -> str:
    value = str(api_mode or "").strip().lower()
    if not value:
        raise ValueError("api_mode is required")
    return value


__all__ = [
    "available_transports",
    "clear_transports_for_tests",
    "get_transport",
    "register_transport",
]
