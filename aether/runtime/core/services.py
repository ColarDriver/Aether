"""Service container for AgentEngine dependencies."""

from __future__ import annotations

import logging
from typing import Optional

from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.memory import MemoryProvider, RetrievalMemoryProvider
from aether.models.provider.base import ModelProvider
from aether.runtime.recovery.fallback_chain import FallbackChain
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.recovery.strategies import RecoveryStrategy
from aether.runtime.control.steer import SteerInbox
from aether.tools.registry import ToolRegistry


class EngineServices:
    """DI container for everything the run-loop reaches into.

    Adding a new dependency here is a deliberate API decision — the
    intent is that ``EngineServices`` enumerates every collaborator that
    can be swapped out for testing or alternative deployments.

    ``fallback_chain`` is an optional field. When present it replaces
    the constructor-time provider
    as the source of truth for "which provider serves the next call":

    * ``services.provider`` (the *currently active* provider) is the
      property the engine dereferences each iteration.  When a chain
      is configured, the property reads from
      ``fallback_chain.current_provider`` so a recovery strategy that
      calls ``fallback_chain.activate_next()`` immediately rotates the
      next ``provider.generate`` call without any extra plumbing.
    * When no chain is configured, ``services.provider`` is a fixed
      reference assigned at construction time and never changes during
      the engine's life.

    The single point of indirection means existing tests that pass a
    bare provider (no chain) keep working unchanged, while the new
    classifier-aware composite strategy can rotate providers without
    needing a special engine constructor argument.

    NB: this class deliberately does NOT use ``@dataclass(slots=True)``.
    Slots conflict with the ``provider`` property and we want the
    property semantics so existing call sites
    (``self.services.provider.generate(...)``) keep working with no
    diff regardless of whether a chain is configured.
    """

    __slots__ = (
        "_initial_provider",
        "tool_registry",
        "middleware_pipeline",
        "interrupt_controller",
        "logger",
        "recovery_strategy",
        "fallback_chain",
        "steer_inbox",
        "memory_provider",
    )

    def __init__(
        self,
        provider: ModelProvider,
        tool_registry: ToolRegistry,
        middleware_pipeline: MiddlewarePipeline,
        interrupt_controller: InterruptController,
        logger: logging.Logger,
        recovery_strategy: RecoveryStrategy,
        fallback_chain: Optional[FallbackChain] = None,
        steer_inbox: Optional[SteerInbox] = None,
        memory_provider: Optional[MemoryProvider] = None,
    ) -> None:
        self._initial_provider = provider
        self.tool_registry = tool_registry
        self.middleware_pipeline = middleware_pipeline
        self.interrupt_controller = interrupt_controller
        self.logger = logger
        self.recovery_strategy = recovery_strategy
        self.fallback_chain = fallback_chain
        self.steer_inbox = steer_inbox or SteerInbox()
        self.memory_provider = memory_provider or RetrievalMemoryProvider()

    @property
    def provider(self) -> ModelProvider:
        """Return the currently-active provider.

        When a ``FallbackChain`` is configured, this re-reads
        ``current_provider`` each call so a recovery strategy that
        rotated mid-loop is observed by the very next invocation
        without explicit refresh plumbing.  When no chain is set, this
        returns the constructor-time provider.

        The property is intentionally cheap: a chain lookup acquires a
        single ``RLock`` and returns a cached reference; the
        non-chained path returns a stored attribute.  Engines call
        this on every iteration's LLM_CALL — keeping it allocation-
        free matters for hot-loop perf.
        """
        if self.fallback_chain is not None:
            return self.fallback_chain.current_provider
        return self._initial_provider

    def __repr__(self) -> str:        # pragma: no cover - debug aid
        chain_size = self.fallback_chain.size if self.fallback_chain else 0
        return (
            f"EngineServices(provider={self._initial_provider!r}, "
            f"chain_size={chain_size})"
        )
