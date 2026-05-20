"""Context engine contracts for compression and provider projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from aether.runtime.core.contracts import TurnContext


@dataclass(slots=True)
class ContextEngineResult:
    """Result returned by a context engine compression step."""

    messages: list[dict[str, Any]]
    changed: bool = False
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    raw_result: Any | None = None


class ContextEngine(Protocol):
    """Boundary for context compression and provider-bound projection."""

    name: str

    def should_compress_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
    ) -> bool:
        """Return whether preflight compression should be considered."""
        ...

    def compact_preflight(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
        trigger_reason: str,
    ) -> ContextEngineResult:
        """Run a preflight compression attempt."""
        ...

    def apply_provider_projection(
        self,
        messages: list[dict[str, Any]],
        *,
        context: TurnContext,
    ) -> list[dict[str, Any]]:
        """Return provider-bound messages without mutating the canonical list."""
        ...


__all__ = [
    "ContextEngine",
    "ContextEngineResult",
]
