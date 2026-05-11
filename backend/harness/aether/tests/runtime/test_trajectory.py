from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aether.runtime.trajectory import (
    build_trajectory_record,
    convert_to_trajectory_format,
    save_trajectory_record,
)


class TrajectoryTests(unittest.TestCase):
    def test_convert_user_assistant_reasoning_and_tool_calls(self) -> None:
        messages = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "what is 2+3?"},
            {
                "role": "assistant",
                "content": "I will calculate.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "sum",
                            "arguments": {"a": 2, "b": 3},
                        },
                    }
                ],
                "metadata": {"reasoning_content": "Need arithmetic."},
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "sum",
                "content": "5",
            },
            {"role": "assistant", "content": "The answer is 5."},
        ]

        converted = convert_to_trajectory_format(messages)

        self.assertEqual(converted[0], {"from": "human", "value": "what is 2+3?"})
        self.assertEqual(converted[1]["from"], "gpt")
        self.assertIn("<think>\nNeed arithmetic.\n</think>", converted[1]["value"])
        self.assertIn("<tool_call>", converted[1]["value"])
        self.assertIn('"name":"sum"', converted[1]["value"])
        self.assertEqual(converted[2]["from"], "tool")
        self.assertIn("<tool_response>", converted[2]["value"])
        self.assertIn('"tool_call_id":"call-1"', converted[2]["value"])
        self.assertEqual(converted[3], {"from": "gpt", "value": "The answer is 5."})

    def test_consecutive_tool_results_are_merged(self) -> None:
        converted = convert_to_trajectory_format(
            [
                {"role": "user", "content": "run tools"},
                {
                    "role": "tool",
                    "tool_call_id": "a",
                    "name": "first",
                    "content": '{"ok": true}',
                },
                {
                    "role": "tool",
                    "tool_call_id": "b",
                    "name": "second",
                    "content": "plain",
                },
            ]
        )

        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[1]["from"], "tool")
        self.assertEqual(converted[1]["value"].count("<tool_response>"), 2)
        self.assertIn('"content":{"ok":true}', converted[1]["value"])
        self.assertIn('"content":"plain"', converted[1]["value"])

    def test_save_trajectory_record_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = build_trajectory_record(
                messages=[{"role": "user", "content": "hello"}],
                session_id="sess/1",
                turn_id="turn-1",
                task_id="task-1",
                model="model-a",
                provider="provider-a",
                completed=True,
            )

            path = save_trajectory_record(
                record,
                trajectory_dir=Path(tmp),
                session_id="sess/1",
                completed=True,
            )

            self.assertEqual(path.parent.name, "sessions")
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["session_id"], "sess/1")
            self.assertEqual(payload["turn_id"], "turn-1")
            self.assertEqual(payload["task_id"], "task-1")
            self.assertEqual(payload["model"], "model-a")
            self.assertTrue(payload["completed"])


if __name__ == "__main__":
    unittest.main()
