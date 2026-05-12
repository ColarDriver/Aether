"""Sprint 3 / PR 3.4 — token estimator unit tests.

The estimator is intentionally rough (4 chars/token + 4/3 padding) — it
only needs ``"is this prompt above 85 % of the window"`` granularity so
the pipeline can decide whether to compress.  These tests pin the
arithmetic that the rest of the compaction surface depends on:

* Group J — basic per-block accounting (string content, list content
  with text / tool_use / tool_result / image / thinking blocks).
* Group K — determinism + monotonicity.  Two consecutive calls on the
  same input MUST return the same number, and removing a message must
  never *increase* the estimate.  These two invariants together let the
  pipeline trust the estimator's "tier_after <= target" check.
"""

from __future__ import annotations

import json
import unittest

from aether.services.compact.token_estimation import (
    _estimate_block_chars,
    estimate_messages_tokens,
)


def _expected(total_chars: int) -> int:
    """Mirror the estimator's ``int((chars / 4) * (4 / 3))`` formula."""
    return int((total_chars / 4) * (4 / 3))


class TokenEstimationGroupJBasicsTests(unittest.TestCase):
    """Group J — basic per-block accounting."""

    def test_j1_empty_messages_returns_zero(self) -> None:
        """Empty list has nothing to count — must return exactly 0."""
        self.assertEqual(estimate_messages_tokens([]), 0)

    def test_j2_single_short_user_message(self) -> None:
        """One short message: overhead 4 + len('hello')=5 → 9 chars."""
        msgs = [{"role": "user", "content": "hello"}]
        self.assertEqual(estimate_messages_tokens(msgs), _expected(4 + 5))

    def test_j3_single_long_user_message(self) -> None:
        """Long English text scales linearly: 1000 chars + 4 overhead."""
        body = "a" * 1000
        msgs = [{"role": "user", "content": body}]
        self.assertEqual(estimate_messages_tokens(msgs), _expected(4 + 1000))

    def test_j4_image_block_uses_8000_char_heuristic(self) -> None:
        """Image blocks count as ~2000 tokens regardless of payload."""
        msgs = [{"role": "user", "content": [{"type": "image", "source": {"x": 1}}]}]
        self.assertEqual(estimate_messages_tokens(msgs), _expected(4 + 8000))

    def test_j5_tool_use_block_is_overhead_plus_input_json(self) -> None:
        """tool_use: 20 char overhead + len(json.dumps(input))."""
        payload = {"path": "/tmp/x", "limit": 100}
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "read_file", "input": payload},
                ],
            }
        ]
        expected_chars = 4 + 20 + len(json.dumps(payload, ensure_ascii=False))
        self.assertEqual(estimate_messages_tokens(msgs), _expected(expected_chars))

    def test_j6_thinking_block_overhead_plus_text(self) -> None:
        """thinking: 20 overhead + len(thinking)."""
        body = "let me think about this for a moment"
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": body}],
            }
        ]
        self.assertEqual(estimate_messages_tokens(msgs), _expected(4 + 20 + len(body)))

    def test_j7_mixed_block_list_sums_per_block(self) -> None:
        """Mixed content: every block type accounted for additively."""
        text = "summary text" * 20
        tool_input = {"q": "hello"}
        tool_inner = "the file contents"
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "tool_use", "id": "c1", "name": "search", "input": tool_input},
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": tool_inner,
                    },
                ],
            }
        ]
        expected_chars = (
            4
            + len(text)
            + 20
            + len(json.dumps(tool_input, ensure_ascii=False))
            + 20
            + len(tool_inner)
        )
        self.assertEqual(estimate_messages_tokens(msgs), _expected(expected_chars))

    def test_j8_tool_result_with_block_list_recurses(self) -> None:
        """Nested tool_result content (list of blocks) recurses through ``_estimate_block_chars``."""
        inner_text = "inner text body"
        msgs = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": [{"type": "text", "text": inner_text}],
                    }
                ],
            }
        ]
        expected_chars = 4 + 20 + len(inner_text)
        self.assertEqual(estimate_messages_tokens(msgs), _expected(expected_chars))

    def test_j9_unknown_block_type_falls_back_to_json(self) -> None:
        """Unknown block types serialise to JSON for length accounting."""
        custom = {"type": "custom", "payload": "abc"}
        msgs = [{"role": "user", "content": [custom]}]
        expected_chars = 4 + len(json.dumps(custom, ensure_ascii=False))
        self.assertEqual(estimate_messages_tokens(msgs), _expected(expected_chars))

    def test_j10_non_dict_message_handled_gracefully(self) -> None:
        """Estimator must not crash on garbage messages — counts the str()."""
        msgs = ["raw string", 42]  # type: ignore[list-item]
        self.assertEqual(
            estimate_messages_tokens(msgs),  # type: ignore[arg-type]
            _expected(len("raw string") + len("42")),
        )

    def test_j11_dict_content_falls_through_to_json_dump(self) -> None:
        """If ``content`` is a bare dict, serialise + count its JSON length."""
        content = {"foo": "bar"}
        msgs = [{"role": "user", "content": content}]
        expected_chars = 4 + len(json.dumps(content, ensure_ascii=False))
        self.assertEqual(estimate_messages_tokens(msgs), _expected(expected_chars))


class TokenEstimationGroupKConsistencyTests(unittest.TestCase):
    """Group K — determinism + monotonicity invariants."""

    def _build_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "do the thing"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "okay" * 30},
                    {"type": "tool_use", "id": "c1", "name": "n", "input": {"a": 1}},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "out" * 50}
                ],
            },
        ]

    def test_k1_repeat_calls_return_same_number(self) -> None:
        """Estimator must be deterministic so pipeline math is stable."""
        msgs = self._build_messages()
        first = estimate_messages_tokens(msgs)
        for _ in range(9):
            self.assertEqual(estimate_messages_tokens(msgs), first)

    def test_k2_dropping_message_never_increases_estimate(self) -> None:
        """Removing the last message must never *grow* the estimate."""
        msgs = self._build_messages()
        full = estimate_messages_tokens(msgs)
        shorter = estimate_messages_tokens(msgs[:-1])
        self.assertLessEqual(shorter, full)
        self.assertLess(shorter, full)  # at least one message gone — should shrink

    def test_k3_estimator_is_pure_does_not_mutate_input(self) -> None:
        """Estimator must not mutate its argument list (pipeline reuses it)."""
        msgs = self._build_messages()
        snapshot = json.dumps(msgs, ensure_ascii=False, default=str)
        estimate_messages_tokens(msgs)
        self.assertEqual(json.dumps(msgs, ensure_ascii=False, default=str), snapshot)

    def test_k4_block_helper_handles_non_dict_block_types(self) -> None:
        """Defensive: a stringy block falls through to ``len(str(...))``."""
        self.assertEqual(_estimate_block_chars("plain text"), len("plain text"))
        self.assertEqual(_estimate_block_chars(42), len("42"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
