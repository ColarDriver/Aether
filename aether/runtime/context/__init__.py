"""Context compression and provider projection runtime boundaries."""

from aether.runtime.context.default_engine import (
    DefaultContextEngine,
    DefaultContextEngineAdapter,
)
from aether.runtime.context.engine import ContextEngine, ContextEngineResult

__all__ = [
    "ContextEngine",
    "ContextEngineResult",
    "DefaultContextEngine",
    "DefaultContextEngineAdapter",
]
