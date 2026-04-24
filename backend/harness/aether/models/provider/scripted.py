"""Testing provider that returns scripted responses."""

from __future__ import annotations

from collections import deque
from typing import Iterable, List

from aether.config.schema import ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import NormalizedResponse, StreamDeltaCallback, TurnContext
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
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        if not self._responses:
            raise RuntimeError("ScriptedProvider has no remaining responses")
        response = self._responses.popleft()
        if stream_callback and response.content and not response.tool_calls:
            try:
                stream_callback(response.content)
            except Exception:  # pragma: no cover - defensive callback path
                pass
        return response
