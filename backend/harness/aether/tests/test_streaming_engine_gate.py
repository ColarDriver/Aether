"""Sprint 1 / PR 1.1 — engine-level streaming gate.

When ``EngineConfig.streaming_enabled`` is False, the engine must NOT
forward ``request.stream_callback`` to the provider, even if the user
supplied one.  This is the emergency rollback switch operators flip when
a gateway's SSE is broken — graceful degradation to one-shot output.
"""

from __future__ import annotations

import unittest
from typing import List

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    StreamDeltaCallback,
    TurnContext,
)
from aether.tools.base import ToolDescriptor


class _CallbackSpyProvider(ModelProvider):
    """Records whether stream_callback was non-None when generate was called."""

    def __init__(self, response: NormalizedResponse) -> None:
        self._response = response
        self.callback_was_present_calls: list[bool] = []

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
    ) -> NormalizedResponse:
        self.callback_was_present_calls.append(stream_callback is not None)
        if stream_callback is not None and self._response.content:
            stream_callback(self._response.content)
        return self._response


class StreamingGateTests(unittest.TestCase):
    def test_default_forwards_callback_to_provider(self) -> None:
        # Sanity: with streaming_enabled=True (default) the provider sees
        # a non-None stream_callback.
        provider = _CallbackSpyProvider(NormalizedResponse(content="ok"))
        engine = AgentEngine(provider)
        deltas: list[str] = []

        result = engine.run_turn(
            EngineRequest(session_id="s", user_message="hi", stream_callback=deltas.append)
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(provider.callback_was_present_calls, [True])
        self.assertEqual(deltas, ["ok"])

    def test_disabled_streaming_suppresses_provider_callback(self) -> None:
        # With streaming_enabled=False the engine must hand the provider
        # ``None`` even though the request had a callback.  The provider
        # therefore takes its non-streaming path and the user STILL gets
        # to see the final content via the engine's "fallback emit one
        # final chunk" path — but only once, and only after generation.
        provider = _CallbackSpyProvider(NormalizedResponse(content="hello"))
        engine = AgentEngine(provider, config=EngineConfig(streaming_enabled=False))
        deltas: list[str] = []

        result = engine.run_turn(
            EngineRequest(session_id="s", user_message="hi", stream_callback=deltas.append)
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        # Provider saw stream_callback=None — the gate worked.
        self.assertEqual(provider.callback_was_present_calls, [False])
        # User should NOT have received a streamed delta when the gate
        # blocks streaming end-to-end (the engine's fallback emit path
        # is gated on stream_callback_wrapped being non-None, which it
        # also isn't when streaming_enabled=False).
        self.assertEqual(deltas, [])

    def test_disabled_streaming_does_not_break_text_completion(self) -> None:
        # Whether or not streaming is on, a text response must still
        # land in result.final_response.
        provider = _CallbackSpyProvider(NormalizedResponse(content="content"))
        engine = AgentEngine(provider, config=EngineConfig(streaming_enabled=False))

        result = engine.run_turn(
            EngineRequest(session_id="s", user_message="hi", stream_callback=lambda _d: None)
        )

        self.assertEqual(result.final_response, "content")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
