"""Provider transport contracts and registry.

Transports own provider-specific payload conversion and response projection.
They do not own HTTP clients, credentials, retries, streaming IO, or engine
recovery policy.
"""

from aether.models.transport.base import ProviderTransport
from aether.models.transport.registry import (
    available_transports,
    clear_transports_for_tests,
    get_transport,
    register_transport,
)
from aether.models.transport.types import TransportPayload, TransportValidation

__all__ = [
    "ProviderTransport",
    "TransportPayload",
    "TransportValidation",
    "available_transports",
    "clear_transports_for_tests",
    "get_transport",
    "register_transport",
]
