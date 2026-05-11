from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.models.provider.base import ModelProvider
from aether.runtime.contracts import EngineRequest, NormalizedResponse, StreamDeltaCallback, StreamSilentCallback, TurnContext
from aether.runtime.interrupts import InterruptController


class _InterruptingProvider(ModelProvider):
    def __init__(self, controller: InterruptController) -> None:
        self._controller = controller

    def generate(
        self,
        messages: list[dict],
        tools,
        config,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        assert stream_callback is not None
        stream_callback("hello")
        self._controller.request(context.session_id, reason="user-interrupt")
        stream_callback(" world")
        return NormalizedResponse(content="ignored")


class InterruptMessageInjectionTests(unittest.TestCase):
    def test_partial_and_marker_are_preserved(self) -> None:
        controller = InterruptController()
        engine = AgentEngine(_InterruptingProvider(controller), interrupt_controller=controller)
        result = engine.run_turn(EngineRequest(session_id="s1", user_message="hi"))
        self.assertEqual(result.status.value, "INTERRUPTED")
        self.assertGreaterEqual(len(result.messages), 3)
        self.assertEqual(result.messages[-2]["role"], "assistant")
        self.assertEqual(result.messages[-2]["content"], "hello")
        self.assertEqual(result.messages[-1]["role"], "user")
        self.assertIn("[Request interrupted by user]", result.messages[-1]["content"])
        interrupt_meta = result.metadata.get("interrupt") or {}
        self.assertEqual(interrupt_meta.get("marker"), "[Request interrupted by user]")
        self.assertEqual(interrupt_meta.get("partial_assistant_chars"), 5)


if __name__ == "__main__":
    unittest.main()
