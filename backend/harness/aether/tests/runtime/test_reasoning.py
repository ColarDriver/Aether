from __future__ import annotations

import unittest

from aether.runtime.observability.reasoning import extract_last_reasoning


class ReasoningExtractionTests(unittest.TestCase):
    def test_extracts_latest_non_empty_reasoning_in_current_turn(self) -> None:
        messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old", "metadata": {"reasoning_content": "old reasoning"}},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "first", "metadata": {"reasoning_content": "first reasoning"}},
            {"role": "assistant", "content": "final"},
        ]

        self.assertEqual(extract_last_reasoning(messages, 2), "first reasoning")

    def test_does_not_cross_user_turn_boundary(self) -> None:
        messages = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old", "metadata": {"reasoning_content": "old reasoning"}},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "final"},
        ]

        self.assertIsNone(extract_last_reasoning(messages, 2))

    def test_reasoning_details_are_stable_compact_json(self) -> None:
        messages = [
            {"role": "user", "content": "new"},
            {
                "role": "assistant",
                "content": "answer",
                "metadata": {"reasoning_details": [{"type": "text", "text": "work"}]},
            },
        ]

        self.assertEqual(
            extract_last_reasoning(messages, 0),
            '[{"type":"text","text":"work"}]',
        )

    def test_top_level_reasoning_fields_are_supported(self) -> None:
        messages = [
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "answer", "reasoning": "top level"},
        ]

        self.assertEqual(extract_last_reasoning(messages, 0), "top level")


if __name__ == "__main__":
    unittest.main()
