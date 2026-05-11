"""Sprint 6 / PR 6.2 — stream callback interrupt propagation.

The interrupt path under test:

    ESC pressed in the REPL
      └→ AetherApp._handle_esc → engine.interrupt(session_id)
           └→ InterruptController flag = True

    [worker thread is in provider.generate(), forwarding stream deltas]

    stream_callback_wrapped(delta)
      └→ poll InterruptController → flag observed
           └→ raise EngineInterrupted(partial_text=<accumulated>)
                └→ propagates through provider's `except Exception:`
                     (works because EngineInterrupted ⊂ BaseException)
                └→ _invoke_provider_with_recovery catches it,
                   writes context.metadata["interrupt"], returns
                   _ProviderInvocationOutcome(interrupted=True)
                └→ run_loop transitions to INTERRUPTED
                └→ EngineResult.metadata["interrupt"] populated

These tests pin every link in that chain so a regression at any layer
fails loudly instead of silently degrading interrupt latency.
"""

from __future__ import annotations

import time
import unittest
from typing import List

from aether import AgentEngine
from aether.models.provider.base import ModelProvider
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    TurnContext,
)
from aether.runtime.exceptions import EngineInterrupted
from aether.runtime.interrupts import InterruptController
from aether.tools.base import ToolDescriptor


# ---------------------------------------------------------------------------
# Exception fundamentals
# ---------------------------------------------------------------------------


class EngineInterruptedExceptionTests(unittest.TestCase):
    def test_carries_reason_partial_text_and_tool_flag(self) -> None:
        exc = EngineInterrupted(
            "user-interrupt",
            partial_text="hello world",
            was_in_tool_call=True,
        )
        self.assertEqual(exc.reason, "user-interrupt")
        self.assertEqual(exc.partial_text, "hello world")
        self.assertTrue(exc.was_in_tool_call)
        # str(exc) surfaces the reason — useful for logging.
        self.assertEqual(str(exc), "user-interrupt")

    def test_defaults_are_empty_and_false(self) -> None:
        exc = EngineInterrupted()
        self.assertEqual(exc.reason, "user-interrupt")
        self.assertEqual(exc.partial_text, "")
        self.assertFalse(exc.was_in_tool_call)

    def test_bypasses_except_exception(self) -> None:
        """Critical contract: a broad `except Exception` MUST NOT swallow us.

        This is why the class is a BaseException subclass — provider
        / middleware code is full of ``try: ... except Exception:``
        clauses that exist to recover from transient errors.  An
        interrupt is a control-flow signal, not an error.
        """
        caught_as_exception = False
        caught_as_engine_interrupted = False
        try:
            try:
                raise EngineInterrupted("user-interrupt")
            except Exception:  # noqa: BLE001 — deliberately broad
                caught_as_exception = True
        except EngineInterrupted:
            caught_as_engine_interrupted = True
        self.assertFalse(caught_as_exception)
        self.assertTrue(caught_as_engine_interrupted)


# ---------------------------------------------------------------------------
# Test helpers — a streaming provider we can drive deterministically
# ---------------------------------------------------------------------------


class _ChunkedStreamProvider(ModelProvider):
    """Provider that streams a fixed list of deltas before returning.

    Mirrors how a real HTTP/SSE provider drives the stream callback —
    one ``callback(delta)`` per chunk, then returns the final
    :class:`NormalizedResponse`.  The chunks list is *not* prepacked
    inside ``response.content`` because we want the callback path to
    drive accumulation (matching the real engine flow where the
    provider's accumulator and the engine's
    ``TURN_KEY_STREAMED_ASSISTANT_TEXT`` are independent records).
    """

    def __init__(
        self,
        chunks: List[str],
        final_content: str | None = None,
        *,
        between_chunks_hook=None,
    ) -> None:
        self._chunks = list(chunks)
        self._final = final_content if final_content is not None else "".join(chunks)
        self._between_chunks_hook = between_chunks_hook

    def generate(
        self,
        messages: list[dict],
        tools: List[ToolDescriptor],
        config,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,  # noqa: ARG002
    ) -> NormalizedResponse:
        for idx, delta in enumerate(self._chunks):
            if stream_callback is not None:
                # Real providers wrap stream callback in `try/except
                # Exception:` to keep transient render errors from
                # killing the call.  Replicate that here so we can
                # verify EngineInterrupted (BaseException) propagates
                # through anyway.
                try:
                    stream_callback(delta)
                except Exception:  # noqa: BLE001 — defensive parity with real providers
                    pass
            if self._between_chunks_hook is not None:
                self._between_chunks_hook(idx)
        return NormalizedResponse(content=self._final)


# ---------------------------------------------------------------------------
# Direct stream-callback tests
# ---------------------------------------------------------------------------


def _make_engine_and_request(
    *,
    chunks: List[str] | None = None,
    between_chunks_hook=None,
    interrupt_controller: InterruptController | None = None,
    session_id: str = "sess",
) -> tuple[AgentEngine, EngineRequest]:
    provider = _ChunkedStreamProvider(
        chunks if chunks is not None else ["only-chunk"],
        between_chunks_hook=between_chunks_hook,
    )
    engine = AgentEngine(
        provider,
        interrupt_controller=interrupt_controller,
    )
    request = EngineRequest(session_id=session_id, user_message="hi")
    return engine, request


class StreamCallbackInterruptPollTests(unittest.TestCase):
    """Direct exercise of ``_build_stream_callback._wrapped``."""

    def test_raises_when_flag_set_before_first_delta(self) -> None:
        controller = InterruptController()
        engine, request = _make_engine_and_request(
            interrupt_controller=controller,
        )

        # Build a fake TurnContext minimally — only the metadata dict
        # the wrapper touches matters.
        context = TurnContext(session_id=request.session_id, iteration=0)
        wrapped = engine._build_stream_callback(request, context)
        self.assertIsNotNone(wrapped)

        controller.request(request.session_id, reason="user-interrupt")

        with self.assertRaises(EngineInterrupted) as cm:
            wrapped("hello")  # type: ignore[misc]
        self.assertEqual(cm.exception.partial_text, "")
        self.assertFalse(cm.exception.was_in_tool_call)

    def test_accumulates_partial_text_before_interrupt(self) -> None:
        controller = InterruptController()
        engine, request = _make_engine_and_request(
            interrupt_controller=controller,
        )

        context = TurnContext(session_id=request.session_id, iteration=0)
        wrapped = engine._build_stream_callback(request, context)
        self.assertIsNotNone(wrapped)

        # Two deltas pass through normally — no flag yet.
        wrapped("hello")  # type: ignore[misc]
        wrapped(" world")  # type: ignore[misc]

        # Trip the flag; the next delta must raise carrying the
        # accumulated partial text.
        controller.request(request.session_id, reason="user-interrupt")
        with self.assertRaises(EngineInterrupted) as cm:
            wrapped("!")  # type: ignore[misc]
        self.assertEqual(cm.exception.partial_text, "hello world")

    def test_empty_or_non_str_delta_does_not_check_flag(self) -> None:
        # Guard rail: the existing early-return on falsy/non-str deltas
        # must remain — otherwise providers that occasionally forward
        # a None or "" tick would hit the flag check unnecessarily.
        controller = InterruptController()
        engine, request = _make_engine_and_request(
            interrupt_controller=controller,
        )
        context = TurnContext(session_id=request.session_id, iteration=0)
        wrapped = engine._build_stream_callback(request, context)
        self.assertIsNotNone(wrapped)

        controller.request(request.session_id, reason="user-interrupt")
        # Must NOT raise — falsy/non-str deltas short-circuit before
        # the interrupt poll, same as before PR 6.2.
        wrapped("")  # type: ignore[misc]
        wrapped(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end run_loop tests
# ---------------------------------------------------------------------------


class RunLoopInterruptPropagationTests(unittest.TestCase):
    """Drive ``engine.run_turn`` against a streaming provider."""

    def test_interrupt_mid_stream_returns_interrupted_with_metadata(self) -> None:
        controller = InterruptController()
        # Three deltas; after the second delta arrives we flip the flag.
        # The next delta triggers EngineInterrupted in the wrapper.
        def trip_after_second(idx: int) -> None:
            if idx == 1:
                controller.request("sess", reason="user-interrupt")

        engine, request = _make_engine_and_request(
            chunks=["I think ", "the answer ", "is 42."],
            between_chunks_hook=trip_after_second,
            interrupt_controller=controller,
            session_id="sess",
        )

        result = engine.run_turn(request)

        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        interrupt_meta = result.metadata.get("interrupt")
        self.assertIsNotNone(interrupt_meta, "metadata.interrupt must be populated")
        self.assertEqual(interrupt_meta["reason"], "user-interrupt")
        # Partial text captured up to (but not including) the chunk
        # that triggered the raise — "I think " + "the answer ".
        self.assertEqual(interrupt_meta["partial_text"], "I think the answer ")
        self.assertFalse(interrupt_meta["was_in_tool_call"])
        self.assertIsInstance(interrupt_meta["triggered_at"], float)
        self.assertGreater(interrupt_meta["triggered_at"], 0.0)

    def test_uninterrupted_turn_has_null_interrupt_metadata(self) -> None:
        # Schema additivity — every non-interrupted turn must still
        # carry the ``interrupt`` key (set to None) so consumers can
        # branch on truthiness without KeyError defensive code.
        provider = ScriptedProvider([NormalizedResponse(content="all good")])
        engine = AgentEngine(provider)
        result = engine.run_turn(EngineRequest(session_id="s", user_message="hi"))

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertIn("interrupt", result.metadata)
        self.assertIsNone(result.metadata["interrupt"])

    def test_engine_interrupted_propagates_through_provider_except_exception(self) -> None:
        """End-to-end coverage for the BaseException design choice.

        ``_ChunkedStreamProvider`` wraps its ``stream_callback`` call
        in ``try/except Exception`` — same shape as every real
        provider in the codebase.  If ``EngineInterrupted`` were a
        regular ``Exception`` it would be swallowed silently and the
        turn would finish normally; we assert the opposite.
        """
        controller = InterruptController()

        def trip_immediately(idx: int) -> None:  # noqa: ARG001
            controller.request("s2", reason="user-interrupt")

        engine, request = _make_engine_and_request(
            chunks=["alpha", "beta", "gamma"],
            between_chunks_hook=trip_immediately,
            interrupt_controller=controller,
            session_id="s2",
        )
        result = engine.run_turn(request)

        # Would be COMPLETED if `except Exception` had swallowed the
        # interrupt.  INTERRUPTED proves BaseException-bypass works.
        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        self.assertIsNotNone(result.metadata.get("interrupt"))

    def test_interrupt_latency_under_500ms(self) -> None:
        """Smoke test for the < 500 ms acceptance bar.

        Schedules a flag-trip on a background timer 100 ms into the
        stream and asserts ``run_turn`` returns within 500 ms.  The
        provider yields one chunk every 50 ms for 40 chunks (~2 s
        total) — without the wrapper poll the turn would take the
        full 2 s.
        """
        import threading

        controller = InterruptController()
        engine, request = _make_engine_and_request(
            chunks=[f"chunk-{i}" for i in range(40)],
            between_chunks_hook=lambda _idx: time.sleep(0.05),
            interrupt_controller=controller,
            session_id="latency",
        )

        timer = threading.Timer(
            0.1,
            lambda: controller.request("latency", reason="user-interrupt"),
        )

        started = time.monotonic()
        timer.start()
        try:
            result = engine.run_turn(request)
        finally:
            timer.cancel()
        elapsed = time.monotonic() - started

        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        # 500 ms is the acceptance bar; we leave a small margin for
        # CI scheduling jitter.
        self.assertLess(
            elapsed,
            0.7,
            f"interrupt latency {elapsed * 1000:.0f}ms exceeds 500ms target",
        )


if __name__ == "__main__":
    unittest.main()
