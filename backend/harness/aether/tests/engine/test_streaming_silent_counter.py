"""Live token counter advances during tool-only streaming phases.

Sprint 3 / PR 3.1 — locks in the contract introduced for the
``stream_silent_callback`` channel.

Background
----------
OpenAI-style chat-completions endpoints stream tool-call argument
fragments via ``delta.tool_calls.function.arguments`` for many
seconds before any visible body content shows up.  Our visible
``stream_callback`` is wired to ``delta.content`` only — so without
the silent channel the activity bar's "↓ N tokens" counter stayed
at zero throughout the entire tool-arg phase, making the user think
the model was hung.

The ``stream_silent_callback`` is a count-only sibling: providers
forward each non-visible chunk to it; the CLI bumps
``_turn_state.response_chars`` without buffering or rendering the
content (mirrors claude-code's ``onUpdateLength`` semantics where
``input_json_delta`` advances the counter independently of the
visible message store).
"""

from __future__ import annotations

import unittest
from typing import Any

from aether.cli.activity import MIN_DISPLAY_TOKENS, TOKEN_CHAR_RATIO, TurnState
from aether.cli.ui import CLIUI
from aether.models.provider.openai_compatible import _parse_sse_stream
from rich.console import Console


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_ui() -> CLIUI:
    """Construct a CLIUI with a captured Rich console (no real stdout)."""
    ui = CLIUI(console=Console(record=True, width=120, file=None))
    return ui


def _wrap_sse(events: list[dict[str, Any]]) -> list[str]:
    """Render OpenAI-style SSE events as ``data: {...}`` lines."""
    import json

    out: list[str] = []
    for ev in events:
        out.append("data: " + json.dumps(ev))
        out.append("")  # SSE message terminator
    out.append("data: [DONE]")
    return out


# ---------------------------------------------------------------------------
# CLIUI.bump_response_chars
# ---------------------------------------------------------------------------


class CLIUIBumpResponseCharsTests(unittest.TestCase):
    def test_bump_advances_response_chars_without_buffering(self) -> None:
        ui = _make_ui()
        # Open a stream so begin_stream's reset doesn't stomp our bump.
        ui.begin_stream()
        # Empty buffer baseline.
        self.assertEqual(ui._stream.char_count, 0)
        self.assertEqual(ui._turn_state.response_chars, 0)

        ui.bump_response_chars('{"path":"/tmp"}')

        # ``response_chars`` advanced — that's what ActivityBar reads.
        self.assertEqual(ui._turn_state.response_chars, len('{"path":"/tmp"}'))
        self.assertEqual(ui._stream.char_count, len('{"path":"/tmp"}'))
        # ... but the visible-body buffer must NOT have received the
        # tool-arg JSON, otherwise the preview / final flush would
        # leak it into the transcript.
        self.assertEqual(ui._stream_buffer, [])

    def test_bump_accumulates_across_calls(self) -> None:
        ui = _make_ui()
        ui.begin_stream()
        for chunk in ['{"', 'path', '":', '"/tmp"', "}"]:
            ui.bump_response_chars(chunk)
        self.assertEqual(ui._turn_state.response_chars, len('{"path":"/tmp"}'))

    def test_bump_ignores_empty_or_non_string(self) -> None:
        ui = _make_ui()
        ui.begin_stream()
        ui.bump_response_chars("")
        ui.bump_response_chars(None)  # type: ignore[arg-type]
        ui.bump_response_chars(123)  # type: ignore[arg-type]
        self.assertEqual(ui._turn_state.response_chars, 0)

    def test_make_silent_callback_returns_callable_that_bumps(self) -> None:
        ui = _make_ui()
        ui.begin_stream()
        cb = ui.make_stream_silent_callback()
        cb("ABCD")
        self.assertEqual(ui._turn_state.response_chars, 4)

    def test_silent_bump_is_enough_to_show_activity_bar_token_counter(self) -> None:
        """End-to-end: silent bumps alone cross MIN_DISPLAY_TOKENS and
        the ↓ N tokens field appears."""
        from aether.cli.activity import ActivityBar

        ui = _make_ui()
        ui.begin_stream()
        # Long enough to clear the floor (≥ ``MIN_DISPLAY_TOKENS`` * 4 chars).
        chars = "x" * (TOKEN_CHAR_RATIO * MIN_DISPLAY_TOKENS + 5)
        ui.bump_response_chars(chars)

        # Render the bar and verify the token field appears.
        ui._turn_state.verb = "Thinking"
        import time as _time
        ui._turn_state.started_at = _time.monotonic() - 3.0  # 3s elapsed
        out_console = Console(record=True, width=80)
        out_console.print(ActivityBar(ui._turn_state))
        out = out_console.export_text()
        self.assertIn("tokens", out)
        self.assertIn("↓", out)


# ---------------------------------------------------------------------------
# OpenAI-compatible SSE parser forwards tool-arg fragments
# ---------------------------------------------------------------------------


class OpenAIToolArgSilentForwardingTests(unittest.TestCase):
    """``_parse_sse_stream`` routes tool-arg deltas to the silent channel."""

    def _stream_events(self) -> list[dict[str, Any]]:
        # A canonical OpenAI tool-call streaming sequence: id+name in
        # the first event, then 4 ``arguments`` fragments.
        return [
            {
                "model": "kimi-k2.6",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "list_dir",
                                        "arguments": '{"',
                                    },
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": "path"},
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '":"/tmp'},
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]

    def test_silent_callback_receives_each_arg_fragment(self) -> None:
        events = self._stream_events()
        visible: list[str] = []
        silent: list[str] = []

        result = _parse_sse_stream(
            iter(_wrap_sse(events)),
            stream_callback=visible.append,
            stream_silent_callback=silent.append,
            fallback_model="kimi-k2.6",
        )

        # Visible channel saw nothing (no delta.content events).
        self.assertEqual(visible, [])
        # Silent channel saw exactly the four argument fragments,
        # in order, untouched.
        self.assertEqual(silent, ['{"', "path", '":"/tmp', '"}'])
        # Final response still parses the ToolCall correctly.
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "list_dir")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "/tmp"})

    def test_silent_callback_optional_keeps_legacy_callers_working(self) -> None:
        events = self._stream_events()
        visible: list[str] = []

        # No silent callback at all — must not raise, must still parse
        # the call out.
        result = _parse_sse_stream(
            iter(_wrap_sse(events)),
            stream_callback=visible.append,
            fallback_model="kimi-k2.6",
        )

        self.assertEqual(visible, [])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].arguments, {"path": "/tmp"})

    def test_silent_callback_exception_does_not_break_stream(self) -> None:
        events = self._stream_events()
        visible: list[str] = []
        silent_seen: list[str] = []

        def _flaky(delta: str) -> None:
            silent_seen.append(delta)
            if delta == "path":
                raise RuntimeError("UI crashed")

        result = _parse_sse_stream(
            iter(_wrap_sse(events)),
            stream_callback=visible.append,
            stream_silent_callback=_flaky,
            fallback_model="kimi-k2.6",
        )

        # All four fragments still visited; the parsed tool call is
        # complete despite the mid-stream raise.
        self.assertEqual(silent_seen, ['{"', "path", '":"/tmp', '"}'])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].arguments, {"path": "/tmp"})

    def test_visible_content_path_does_not_double_emit_to_silent(self) -> None:
        # If the model interleaves visible text (typical for "tools then
        # answer" turns), text deltas must go to ``stream_callback`` ONLY,
        # otherwise the activity bar would double-count those characters.
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hello "},
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_x",
                                    "function": {
                                        "name": "noop",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        visible: list[str] = []
        silent: list[str] = []

        _parse_sse_stream(
            iter(_wrap_sse(events)),
            stream_callback=visible.append,
            stream_silent_callback=silent.append,
            fallback_model="kimi-k2.6",
        )

        self.assertEqual(visible, ["Hello "])
        # The visible "Hello " must NOT have leaked to the silent
        # channel — exactly one event there, the "{}".
        self.assertEqual(silent, ["{}"])


# ---------------------------------------------------------------------------
# End-to-end: provider → silent_callback → CLIUI counter
# ---------------------------------------------------------------------------


class EndToEndSilentCounterTests(unittest.TestCase):
    """Mini integration: feed a synthetic SSE stream into the parser
    with a real CLIUI as the silent sink and assert the activity bar
    would show the counter."""

    def test_full_pipeline_advances_response_chars(self) -> None:
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"/etc/hosts","encoding":"utf-8"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]
        ui = _make_ui()
        ui.begin_stream()

        _parse_sse_stream(
            iter(_wrap_sse(events)),
            stream_callback=ui.make_stream_callback(),
            stream_silent_callback=ui.make_stream_silent_callback(),
            fallback_model="kimi-k2.6",
        )

        expected = len('{"path":"/etc/hosts","encoding":"utf-8"}')
        self.assertEqual(ui._turn_state.response_chars, expected)
        # Visible buffer was not polluted.
        self.assertEqual(ui._stream_buffer, [])


if __name__ == "__main__":
    unittest.main()
