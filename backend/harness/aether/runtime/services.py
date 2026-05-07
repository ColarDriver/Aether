"""Service container for AgentEngine dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.models.provider.base import ModelProvider
from aether.runtime.interrupts import InterruptController
from aether.runtime.recovery import RecoveryStrategy
from aether.tools.registry import ToolRegistry


@dataclass(slots=True)
class EngineServices:
    """DI container for everything the run-loop reaches into.

    Adding a new dependency here is a deliberate API decision — the
    intent is that ``EngineServices`` enumerates every collaborator that
    can be swapped out for testing or alternative deployments.
    """

    provider: ModelProvider
    tool_registry: ToolRegistry
    middleware_pipeline: MiddlewarePipeline
    interrupt_controller: InterruptController
    logger: logging.Logger
    # Sprint 0 / PR 0.3: engine-side retry policy for ProviderInvocationError.
    # See ``runtime/recovery.py`` for the abstraction and shipped strategies.
    recovery_strategy: RecoveryStrategy
