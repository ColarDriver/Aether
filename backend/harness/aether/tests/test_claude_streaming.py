"""Streaming behaviour of :class:`ClaudeChatModel`.

Locks in the contract introduced by Sprint 3 / PR 3.1:

* When ``stream_callback`` is provided, ``generate`` must route through
  ``messages.stream`` rather than the blocking ``messages.create`` —
  otherwise ``response_chars`` stays at zero on the CLI side and the
  activity bar's "↓ N tokens" counter never ticks.
* Each non-empty text chunk is forwarded to the callback as it
  arrives; the legacy "full content" fallback is **suppressed** when
  streaming actually produced text (otherwise the bar would
  double-count every visible character).
* Tool-use turns still honour the streaming path — the SDK's
  ``get_final_message`` returns the same shape as the blocking call,
  so ``_parse_response`` continues to surface ``tool_calls``
  correctly.
* When ``stream_callback`` is ``None`` (e.g. an internal evaluator,
  background agent) we keep the blocking path so we don't pay the
  SSE setup cost where it brings no value.
"""

from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest import mock

# Anthropic SDK constructor refuses to initialize without a key.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from aether.config.schema import ModelCallConfig
from aether.models.provider.claude import ClaudeChatModel
from aether.runtime.contracts import TurnContext


def _make_context() -> TurnContext:
    """Minimum viable ``TurnContext`` for ``generate`` calls."""
    return TurnContext(
        session_id="sess-claude-stream",
        iteration=1,
        metadata={},
    )


def _make_call_config(model: str = "claude-sonnet-4-6") -> ModelCallConfig:
    return ModelCallConfig(
        max_tokens=128,
        temperature=0.0,
        extra={"model": model},
    )


class _FakeStream:
    """Stand-in for the Anthropic SDK's stream context manager.

    Implements the subset we touch: context-manager protocol,
    ``text_stream`` iterator, and ``get_final_message``.  Lets us
    verify forwarding behaviour without a real network round-trip.
    """

    def __init__(self, chunks: list[str], final: dict[str, Any]) -> None:
        self._chunks = chunks
        self._final = final
        self.entered = False
        self.exited = False

    def __enter__(self) -> "_FakeStream":
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.exited = True
        return False

    @property
    def text_stream(self) -> Iterator[str]:
        return iter(self._chunks)

    def get_final_message(self) -> dict[str, Any]:
        return self._final


def _build_model_with_fakes(
    *,
    chunks: list[str],
    final: dict[str, Any],
) -> tuple[ClaudeChatModel, _FakeStream, mock.MagicMock]:
    """Construct a real ``ClaudeChatModel`` wrapped around a fake client."""

    fake_stream = _FakeStream(chunks, final)
    fake_create = mock.MagicMock()

    @contextmanager
    def _open_stream(**_kwargs):
        yield from [fake_stream.__enter__()]

    fake_messages = SimpleNamespace(
        # Use the real context-manager protocol path: the production
        # code does ``with self._client.messages.stream(...) as stream``.
        # ``messages.stream`` itself returns the context manager —
        # not a generator — so we hand back the fake directly.
        stream=mock.MagicMock(return_value=fake_stream),
        create=fake_create,
    )
    fake_client = SimpleNamespace(messages=fake_messages)

    with mock.patch.object(
        ClaudeChatModel,
        "_resolve_api_key",
        return_value="sk-test",
    ), mock.patch.object(
        ClaudeChatModel,
        "_build_client",
        return_value=fake_client,
    ):
        model = ClaudeChatModel(
            model="claude-sonnet-4-6",
            max_tokens=128,
            enable_prompt_caching=False,
            auto_thinking_budget=False,
            retry_max_attempts=1,
        )

    return model, fake_stream, fake_create


class ClaudeStreamingForwardingTests(unittest.TestCase):
    """Each text chunk reaches the callback as it arrives."""

    def test_streams_each_text_chunk_to_callback(self) -> None:
        chunks = ["Hello", ", ", "world", "!"]
        final = {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 4},
        }
        model, fake_stream, fake_create = _build_model_with_fakes(
            chunks=chunks, final=final
        )

        received: list[str] = []
        result = model.generate(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            config=_make_call_config(),
            context=_make_context(),
            stream_callback=received.append,
        )

        self.assertEqual(received, chunks)
        self.assertEqual(result.content, "Hello, world!")
        self.assertEqual(result.tool_calls, [])
        self.assertTrue(fake_stream.entered)
        self.assertTrue(fake_stream.exited)
        # Blocking ``messages.create`` must not be touched on the
        # streaming path.
        fake_create.assert_not_called()

    def test_does_not_double_emit_final_content_when_streamed(self) -> None:
        chunks = ["streamed body chunk one", " and chunk two"]
        final = {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "streamed body chunk one and chunk two"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 8},
        }
        model, _, _ = _build_model_with_fakes(chunks=chunks, final=final)

        received: list[str] = []
        model.generate(
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            config=_make_call_config(),
            context=_make_context(),
            stream_callback=received.append,
        )

        # Exactly the streamed chunks — no extra "full content" tail
        # appended after the loop, otherwise the activity bar's
        # ``response_chars`` would double.
        self.assertEqual(received, chunks)

    def test_skips_empty_chunks(self) -> None:
        chunks = ["", "real", "", "more"]
        final = {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "realmore"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        model, _, _ = _build_model_with_fakes(chunks=chunks, final=final)

        received: list[str] = []
        model.generate(
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            config=_make_call_config(),
            context=_make_context(),
            stream_callback=received.append,
        )

        self.assertEqual(received, ["real", "more"])

    def test_callback_exception_does_not_break_stream(self) -> None:
        chunks = ["a", "b", "c"]
        final = {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "abc"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        model, fake_stream, _ = _build_model_with_fakes(chunks=chunks, final=final)

        seen: list[str] = []

        def _flaky(delta: str) -> None:
            seen.append(delta)
            if delta == "b":
                raise RuntimeError("ui crashed mid-stream")

        result = model.generate(
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            config=_make_call_config(),
            context=_make_context(),
            stream_callback=_flaky,
        )

        # All three chunks are still visited (we keep draining), and
        # the final message arrives intact.
        self.assertEqual(seen, ["a", "b", "c"])
        self.assertEqual(result.content, "abc")
        self.assertTrue(fake_stream.exited)


class ClaudeStreamingFallbackTests(unittest.TestCase):
    """Edge cases around no-stream and tool-use paths."""

    def test_no_callback_uses_blocking_create(self) -> None:
        """Internal callers without a UI take the blocking path."""
        final = {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "blocked"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        chunks: list[str] = []  # not consulted on this path
        model, fake_stream, fake_create = _build_model_with_fakes(
            chunks=chunks, final=final
        )
        fake_create.return_value = final

        result = model.generate(
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            config=_make_call_config(),
            context=_make_context(),
            stream_callback=None,
        )

        self.assertEqual(result.content, "blocked")
        fake_create.assert_called_once()
        self.assertFalse(fake_stream.entered)

    def test_fallback_emits_when_streaming_yields_nothing(self) -> None:
        """If the SDK closed the stream with zero text chunks (e.g.
        a pure tool-only response with no visible text), we fall back
        to forwarding the parsed content once so the UI still sees
        something — provided there are no tool calls."""
        final = {
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "fallback body"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        model, _, _ = _build_model_with_fakes(chunks=[], final=final)

        received: list[str] = []
        model.generate(
            messages=[{"role": "user", "content": "go"}],
            tools=[],
            config=_make_call_config(),
            context=_make_context(),
            stream_callback=received.append,
        )

        self.assertEqual(received, ["fallback body"])

    def test_tool_use_response_streams_then_returns_tool_calls(self) -> None:
        """A turn that produces tool_use must still parse the
        ``tool_use`` block correctly after streaming."""
        chunks = ["Let me check that file."]
        final = {
            "model": "claude-sonnet-4-6",
            "content": [
                {"type": "text", "text": "Let me check that file."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "read_file",
                    "input": {"path": "/etc/hosts"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 6},
        }
        model, _, _ = _build_model_with_fakes(chunks=chunks, final=final)

        received: list[str] = []
        result = model.generate(
            messages=[{"role": "user", "content": "read /etc/hosts"}],
            tools=[],
            config=_make_call_config(),
            context=_make_context(),
            stream_callback=received.append,
        )

        self.assertEqual(received, ["Let me check that file."])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "/etc/hosts"})
        # Even with a tool call, we don't emit the legacy "full
        # content" fallback after streaming — preventing the bar from
        # double-counting the visible portion.
        self.assertEqual(received, ["Let me check that file."])


if __name__ == "__main__":
    unittest.main()
