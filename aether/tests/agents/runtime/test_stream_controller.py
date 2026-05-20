from __future__ import annotations

import logging
import unittest

from aether.agents.middlewares.pipeline import MiddlewarePipeline
from aether.agents.runtime.stream_controller import StreamController
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.control.interrupts import InterruptController
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse, TurnContext
from aether.runtime.core.exceptions import EngineInterrupted
from aether.runtime.core.services import EngineServices
from aether.runtime.recovery.strategies import NoRetryStrategy
from aether.runtime.session.session_runtime import TURN_KEY_STREAMED_ASSISTANT_TEXT
from aether.tools.registry import ToolRegistry


class _Adapter:
    def __init__(self) -> None:
        self.interrupted = False

    def is_interrupted(self, session_id: str, context: TurnContext | None = None) -> bool:
        del session_id, context
        return self.interrupted


def _services() -> EngineServices:
    return EngineServices(
        provider=ScriptedProvider([NormalizedResponse(content="ok")]),
        tool_registry=ToolRegistry(),
        middleware_pipeline=MiddlewarePipeline(),
        interrupt_controller=InterruptController(),
        logger=logging.getLogger(__name__),
        recovery_strategy=NoRetryStrategy(),
    )


class StreamControllerTests(unittest.TestCase):
    def test_visible_delta_forwarded_once_and_metadata_updated(self) -> None:
        adapter = _Adapter()
        controller = StreamController(
            config=EngineConfig(),
            services=_services(),
            adapter=adapter,
        )
        seen: list[str] = []
        context = TurnContext(session_id="stream", iteration=1)

        callbacks = controller.build_callbacks(
            request=EngineRequest(session_id="stream", stream_callback=seen.append),
            context=context,
        )
        assert callbacks.visible is not None

        callbacks.visible("hello")

        self.assertEqual(seen, ["hello"])
        self.assertEqual(context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT], "hello")
        self.assertTrue(context.metadata["streamed_output"])
        self.assertEqual(context.metadata["stream_callback_calls"], 1)

    def test_visible_callback_without_user_callback_keeps_partial_recovery_buffer(self) -> None:
        adapter = _Adapter()
        controller = StreamController(
            config=EngineConfig(),
            services=_services(),
            adapter=adapter,
        )
        context = TurnContext(session_id="stream-partial", iteration=1)

        visible = controller.build_visible_callback(
            request=EngineRequest(session_id="stream-partial"),
            context=context,
        )
        assert visible is not None
        visible("partial")

        self.assertEqual(context.metadata[TURN_KEY_STREAMED_ASSISTANT_TEXT], "partial")
        self.assertNotIn("stream_callback_calls", context.metadata)

    def test_silent_delta_increments_counter_without_visible_text(self) -> None:
        adapter = _Adapter()
        controller = StreamController(
            config=EngineConfig(),
            services=_services(),
            adapter=adapter,
        )
        seen: list[str] = []
        context = TurnContext(session_id="stream-silent", iteration=1)

        silent = controller.build_silent_callback(
            request=EngineRequest(session_id="stream-silent", stream_silent_callback=seen.append),
            context=context,
        )
        assert silent is not None
        silent('{"cmd":')

        self.assertEqual(seen, ['{"cmd":'])
        self.assertEqual(context.metadata["stream_silent_callback_calls"], 1)
        self.assertNotIn(TURN_KEY_STREAMED_ASSISTANT_TEXT, context.metadata)

    def test_visible_interrupt_raises_engine_interrupted(self) -> None:
        adapter = _Adapter()
        controller = StreamController(
            config=EngineConfig(),
            services=_services(),
            adapter=adapter,
        )
        context = TurnContext(
            session_id="stream-interrupt",
            iteration=1,
            metadata={TURN_KEY_STREAMED_ASSISTANT_TEXT: "hello"},
        )
        visible = controller.build_visible_callback(
            request=EngineRequest(session_id="stream-interrupt", stream_callback=lambda _d: None),
            context=context,
        )
        assert visible is not None
        adapter.interrupted = True

        with self.assertRaises(EngineInterrupted) as cm:
            visible(" world")

        self.assertEqual(cm.exception.partial_text, "hello")
        self.assertFalse(cm.exception.was_in_tool_call)

    def test_silent_interrupt_raises_engine_interrupted(self) -> None:
        adapter = _Adapter()
        controller = StreamController(
            config=EngineConfig(),
            services=_services(),
            adapter=adapter,
        )
        context = TurnContext(session_id="stream-silent-interrupt", iteration=1)
        silent = controller.build_silent_callback(
            request=EngineRequest(
                session_id="stream-silent-interrupt",
                stream_silent_callback=lambda _d: None,
            ),
            context=context,
        )
        assert silent is not None
        adapter.interrupted = True

        with self.assertRaises(EngineInterrupted):
            silent("hidden")

    def test_streaming_disabled_suppresses_callbacks(self) -> None:
        adapter = _Adapter()
        controller = StreamController(
            config=EngineConfig(streaming_enabled=False),
            services=_services(),
            adapter=adapter,
        )
        context = TurnContext(session_id="stream-off", iteration=1)

        callbacks = controller.build_callbacks(
            request=EngineRequest(
                session_id="stream-off",
                stream_callback=lambda _d: None,
                stream_silent_callback=lambda _d: None,
            ),
            context=context,
        )

        self.assertIsNone(callbacks.visible)
        self.assertIsNone(callbacks.silent)


if __name__ == "__main__":
    unittest.main()
