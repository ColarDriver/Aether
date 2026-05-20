"""Context compression and provider projection runtime boundaries."""

from aether.runtime.context.compression_lifecycle import (
    CompressionLifecycleService,
    CompressionRequest,
    CompressionResult,
    CompressionStatus,
)
from aether.runtime.context.default_engine import (
    DefaultContextEngine,
    DefaultContextEngineAdapter,
)
from aether.runtime.context.engine import ContextEngine, ContextEngineResult

__all__ = [
    "CompressionLifecycleService",
    "CompressionRequest",
    "CompressionResult",
    "CompressionStatus",
    "ContextEngine",
    "ContextEngineResult",
    "DefaultContextEngine",
    "DefaultContextEngineAdapter",
]
