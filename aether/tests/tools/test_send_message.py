"""Tests for the SendMessageTool — PR 10.7."""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.tools.builtins.send_message import SendMessageTool


def _record(task_id: str, *, status: TaskStatus = TaskStatus.RUNNING) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        parent_session_id="s",
        subagent_type="general-purpose",
        prompt="x",
        status=status,
        started_at=time.time(),
        last_heartbeat=time.time(),
    )


def _ctx(*, store: TaskStore | None = None) -> TurnContext:
    metadata: dict = {}
    if store is not None:
        metadata["_task_store"] = store
    return TurnContext(session_id="s", iteration=0, metadata=metadata)


def _execute(tool: SendMessageTool, args: dict, *, ctx: TurnContext | None = None):
    return tool.execute(
        ToolCall(id="c1", name="send_message", arguments=args),
        ctx or _ctx(),
    )


class InputValidationTests(unittest.TestCase):
    def test_missing_to_errors(self) -> None:
        result = _execute(SendMessageTool(), {"message": "hi"})
        self.assertTrue(result.is_error)
        self.assertIn("'to'", result.content)

    def test_missing_message_errors(self) -> None:
        result = _execute(SendMessageTool(), {"to": "t1"})
        self.assertTrue(result.is_error)
        self.assertIn("'message'", result.content)

    def test_blank_message_rejected(self) -> None:
        result = _execute(SendMessageTool(), {"to": "t1", "message": "   "})
        self.assertTrue(result.is_error)

    def test_summary_must_be_string(self) -> None:
        result = _execute(
            SendMessageTool(),
            {"to": "t1", "message": "hi", "summary": 42},
        )
        self.assertTrue(result.is_error)
        self.assertIn("summary", result.content)


class StoreResolutionTests(unittest.TestCase):
    def test_no_store_configured_returns_error(self) -> None:
        result = _execute(SendMessageTool(), {"to": "t1", "message": "hi"})
        self.assertTrue(result.is_error)
        self.assertIn("TaskStore", result.content)

    def test_constructor_store_wins_over_context(self) -> None:
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            store_a = TaskStore(root=Path(a))
            store_a.create(_record("only-in-a"))
            store_b = TaskStore(root=Path(b))
            tool = SendMessageTool(task_store=store_a)
            result = _execute(
                tool, {"to": "only-in-a", "message": "hi"}, ctx=_ctx(store=store_b)
            )
            self.assertFalse(result.is_error, msg=result.content)


class TargetStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.tool = SendMessageTool(task_store=self.store)

    def test_unknown_task_id_errors(self) -> None:
        result = _execute(self.tool, {"to": "nope", "message": "hi"})
        self.assertTrue(result.is_error)
        self.assertIn("unknown task_id", result.content)

    def test_completed_task_rejects_send(self) -> None:
        self.store.create(_record("done"))
        self.store.update_status("done", TaskStatus.COMPLETED)
        result = _execute(self.tool, {"to": "done", "message": "hi"})
        self.assertTrue(result.is_error)
        self.assertIn("completed", result.content)

    def test_killed_task_rejects_send(self) -> None:
        self.store.create(_record("dead"))
        self.store.update_status("dead", TaskStatus.KILLED)
        result = _execute(self.tool, {"to": "dead", "message": "hi"})
        self.assertTrue(result.is_error)


class QueueingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.tool = SendMessageTool(task_store=self.store)
        self.store.create(_record("running"))

    def test_running_task_succeeds_and_enqueues(self) -> None:
        result = _execute(
            self.tool, {"to": "running", "message": "please also check X"}
        )
        self.assertFalse(result.is_error, msg=result.content)
        self.assertEqual(result.metadata["to"], "running")
        self.assertEqual(result.metadata["queued_chars"], len("please also check X"))
        # Drained value matches what we sent.
        drained = self.store.drain_pending_messages("running")
        self.assertEqual(drained, ["please also check X"])

    def test_summary_optional_recorded_in_metadata(self) -> None:
        result = _execute(
            self.tool,
            {"to": "running", "message": "ping", "summary": "5-word check-in"},
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata.get("summary"), "5-word check-in")

    def test_summary_not_persisted_in_pending_payload(self) -> None:
        _execute(
            self.tool,
            {"to": "running", "message": "ping", "summary": "private note"},
        )
        path = Path(self._tmp.name) / "running" / "pending.jsonl"
        line = path.read_text().splitlines()[-1]
        self.assertEqual(json.loads(line)["message"], "ping")

    def test_message_round_trips_unicode(self) -> None:
        unicode_msg = "请也检查 X (与 Y 的边界条件)"
        _execute(self.tool, {"to": "running", "message": unicode_msg})
        drained = self.store.drain_pending_messages("running")
        self.assertEqual(drained, [unicode_msg])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
