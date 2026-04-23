"""Base model provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from aether.config.schema import ModelCallConfig
from aether.runtime.contracts import NormalizedResponse, TurnContext
from aether.tools.base import ToolDescriptor


class ModelProvider(ABC):
    """Provider contract used by AgentEngine."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
    ) -> NormalizedResponse:
        raise NotImplementedError
