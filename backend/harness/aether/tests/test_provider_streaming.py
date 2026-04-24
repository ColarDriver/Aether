from __future__ import annotations

import unittest

from aether.models.provider.openai import CodexChatModel


class ProviderStreamingTests(unittest.TestCase):
    def test_extract_stream_delta_prefers_direct_delta_string(self) -> None:
        event = {"type": "response.output_text.delta", "delta": "Hello"}
        self.assertEqual(CodexChatModel._extract_stream_delta(event), "Hello")

    def test_extract_stream_delta_from_delta_object(self) -> None:
        event = {"type": "response.output_text.delta", "delta": {"text": " world"}}
        self.assertEqual(CodexChatModel._extract_stream_delta(event), " world")

    def test_extract_stream_delta_from_output_text_event_text(self) -> None:
        event = {"type": "response.output_text.delta", "text": "!"}
        self.assertEqual(CodexChatModel._extract_stream_delta(event), "!")

    def test_extract_stream_delta_from_item_text(self) -> None:
        event = {"type": "response.output_item.added", "item": {"text": "item-text"}}
        self.assertEqual(CodexChatModel._extract_stream_delta(event), "item-text")

    def test_extract_stream_delta_from_item_content_blocks(self) -> None:
        event = {
            "type": "response.output_item.done",
            "item": {
                "content": [
                    {"type": "output_text", "text": "A"},
                    {"type": "output_text", "text": "B"},
                ]
            },
        }
        self.assertEqual(CodexChatModel._extract_stream_delta(event), "AB")

    def test_emit_stream_delta_invokes_callback_for_text(self) -> None:
        deltas: list[str] = []
        event = {"type": "response.output_text.delta", "delta": "chunk"}

        CodexChatModel._emit_stream_delta(deltas.append, event)

        self.assertEqual(deltas, ["chunk"])

    def test_emit_stream_delta_ignores_non_text_events(self) -> None:
        deltas: list[str] = []
        event = {"type": "response.completed", "response": {"id": "r_1"}}

        CodexChatModel._emit_stream_delta(deltas.append, event)

        self.assertEqual(deltas, [])


if __name__ == "__main__":
    unittest.main()
