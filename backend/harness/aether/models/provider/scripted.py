"""Testing provider that returns scripted responses."""

from __future__ import annotations

from collections import deque
from typing import Iterable, List

from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import NormalizedResponse, TurnContext
from aether.tools.base import ToolDescriptor


class ScriptedProvider(ModelProvider):
    """Return predefined responses in sequence."""

    def __init__(self, responses: Iterable[NormalizedResponse]) -> None:
        self._responses = deque(responses)

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
    ) -> NormalizedResponse:
        if not self._responses:
            raise RuntimeError("ScriptedProvider has no remaining responses")
        return self._responses.popleft()
