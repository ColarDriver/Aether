"""Tests for gateway approval prompter bridge (PR 5)."""

from __future__ import annotations

import unittest
from unittest import mock

from aether.gateway.handlers.prompter_bridge import GatewayPrompter


class GatewayPrompterBridgeTests(unittest.TestCase):
    def test_confirm_plan_round_trip(self) -> None:
        prompter = GatewayPrompter(
            session_id="ses_1",
            run_id="run_1",
            request_timeout=12.0,
        )
        with mock.patch(
            "aether.gateway.handlers.prompter_bridge.reverse_rpc.call",
            return_value={"kind": "plan", "confirmed": True},
        ) as call:
            result = prompter.confirm_plan(
                "1. edit file",
                plan_path="/tmp/plan.md",
            )
            self.assertTrue(result["confirmed"])

        method, params = call.call_args.args
        self.assertEqual(method, "approval.request")
        self.assertEqual(params["kind"], "plan")
        self.assertEqual(params["session_id"], "ses_1")
        self.assertEqual(params["run_id"], "run_1")
        self.assertEqual(params["plan_text"], "1. edit file")
        self.assertEqual(params["plan_path"], "/tmp/plan.md")
        self.assertEqual(params["deadline_ms"], 12_000)
        self.assertEqual(call.call_args.kwargs["timeout"], 12.0)

    def test_confirm_plan_false(self) -> None:
        prompter = GatewayPrompter(session_id="ses_1", run_id="run_1")
        with mock.patch(
            "aether.gateway.handlers.prompter_bridge.reverse_rpc.call",
            return_value={"kind": "plan", "confirmed": False},
        ):
            result = prompter.confirm_plan("plan")
            self.assertFalse(result["confirmed"])

    def test_ask_questions_maps_schema_and_returns_answers(self) -> None:
        prompter = GatewayPrompter(session_id="ses_1", run_id="run_1")
        questions = [
            {
                "id": "q1",
                "prompt": "Choose target",
                "options": [
                    {"id": "a", "label": "API"},
                    {"id": "b", "label": "CLI"},
                ],
            },
            {"id": "q2", "prompt": "Notes?", "free_text": True},
        ]
        with mock.patch(
            "aether.gateway.handlers.prompter_bridge.reverse_rpc.call",
            return_value={"kind": "questions", "answers": {"q1": "a", "q2": "ok"}},
        ) as call:
            result = prompter.ask_questions(questions, timeout=5.0)

        self.assertEqual(result, {"q1": "a", "q2": "ok"})
        params = call.call_args.args[1]
        self.assertEqual(params["kind"], "questions")
        self.assertEqual(params["deadline_ms"], 5_000)
        self.assertEqual(params["questions"][0]["kind"], "select")
        self.assertEqual(params["questions"][0]["options"], ["API", "CLI"])
        self.assertEqual(params["questions"][1]["kind"], "open")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
