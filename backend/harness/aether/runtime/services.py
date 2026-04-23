"""Service container for AgentEngine dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.models.provider.base import ModelProvider
from aether.runtime.interrupts import InterruptController
from aether.tools.registry import ToolRegistry


@dataclass(slots=True)
class EngineServices:
    provider: ModelProvider
    tool_registry: ToolRegistry
    middleware_pipeline: MiddlewarePipeline
    interrupt_controller: InterruptController
    logger: logging.Logger
