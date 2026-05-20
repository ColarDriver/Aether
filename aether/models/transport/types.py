"""Small data containers shared by provider transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TransportPayload:
    """Provider request body plus optional transport-local metadata."""

    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TransportValidation:
    """Raw response validation result."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


__all__ = ["TransportPayload", "TransportValidation"]
