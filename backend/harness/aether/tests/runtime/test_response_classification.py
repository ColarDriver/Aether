from __future__ import annotations

import unittest

from aether.runtime.core.contracts import NormalizedResponse
from aether.runtime.recovery.response_classification import (
    EmptyKind,
    is_legitimate_empty,
    strip_thinking_tags,
)


class ResponseClassificationTests(unittest.TestCase):
    def test_visible_content_is_not_empty(self) -> None:
        result = is_legitimate_empty(NormalizedResponse(content=" hello "))

        self.assertEqual(result.kind, EmptyKind.NOT_EMPTY)
        self.assertTrue(result.is_success)
        self.assertFalse(result.is_recoverable)
        self.assertEqual(result.visible_text_chars, 5)

    def test_stop_reason_end_turn_is_legitimate_empty(self) -> None:
        result = is_legitimate_empty(
            NormalizedResponse(content="", finish_reason="", metadata={"stop_reason": "end_turn"})
        )

        self.assertEqual(result.kind, EmptyKind.LEGITIMATE_END_TURN)
        self.assertTrue(result.is_success)

    def test_finish_reason_stop_is_legitimate_empty_without_partial_stream(self) -> None:
        result = is_legitimate_empty(NormalizedResponse(content="", finish_reason="stop"))

        self.assertEqual(result.kind, EmptyKind.LEGITIMATE_END_TURN)

    def test_thinking_metadata_classifies_as_thinking_only(self) -> None:
        result = is_legitimate_empty(
            NormalizedResponse(content="", finish_reason="length", metadata={"reasoning_content": "work"})
        )

        self.assertEqual(result.kind, EmptyKind.THINKING_ONLY)
        self.assertTrue(result.has_thinking)

    def test_thinking_tags_are_not_visible_text(self) -> None:
        result = is_legitimate_empty(
            NormalizedResponse(content="<think>work</think>", finish_reason="length")
        )

        self.assertEqual(result.kind, EmptyKind.THINKING_ONLY)
        self.assertEqual(result.visible_text_chars, 0)

    def test_streamed_partial_turns_empty_into_recoverable_bug(self) -> None:
        result = is_legitimate_empty(
            NormalizedResponse(content="", finish_reason="length", metadata={"reasoning_content": "work"}),
            streamed_assistant_text="partial answer",
        )

        self.assertEqual(result.kind, EmptyKind.BUG_EMPTY)
        self.assertTrue(result.has_thinking)
        self.assertTrue(result.has_streamed_partial)
        self.assertTrue(result.is_recoverable)

    def test_length_empty_without_stop_signal_is_bug_empty(self) -> None:
        result = is_legitimate_empty(NormalizedResponse(content="", finish_reason="length"))

        self.assertEqual(result.kind, EmptyKind.BUG_EMPTY)

    def test_strip_thinking_tags_removes_supported_reasoning_markup(self) -> None:
        self.assertEqual(
            strip_thinking_tags("a<think>hidden</think>b<reasoning>x</reasoning>c"),
            "abc",
        )


if __name__ == "__main__":
    unittest.main()
