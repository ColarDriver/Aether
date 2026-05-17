"""Tests for AgentEngine._drain_pending_messages — PR 10.7."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse, TurnContext
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.subagents import SubagentManager


def _record(task_id: str) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        parent_session_id="s",
        subagent_type="general-purpose",
        prompt="x",
        status=TaskStatus.RUNNING,
        started_at=time.time(),
        last_heartbeat=time.time(),
    )


def _make_engine(*, store: TaskStore | None = None) -> AgentEngine:
    return AgentEngine(
        ScriptedProvider([NormalizedResponse(content="ok")]),
        config=EngineConfig(use_builtin_tools=False),
        task_store=store,
        subagent_manager=SubagentManager(),
    )


class ExternalEventQueueTests(unittest.TestCase):
    def test_enqueue_external_event_appends_to_messages(self) -> None:
        engine = _make_engine()
        engine.enqueue_external_event("hello world")
        ctx = TurnContext(session_id="s", iteration=0, metadata={}, task_id=None)
        nudged = engine._drain_pending_messages([], ctx)
        self.assertEqual(len(nudged), 1)
        self.assertEqual(nudged[0]["role"], "user")
        self.assertEqual(nudged[0]["content"], "hello world")
        # Source tags fall to "external" unless the body smells like a notification.
        self.assertEqual(nudged[0]["metadata"]["source"], "external")

    def test_task_notification_tag_detected(self) -> None:
        engine = _make_engine()
        engine.enqueue_external_event(
            "<task-notification><task_id>t1</task_id></task-notification>"
        )
        ctx = TurnContext(session_id="s", iteration=0, metadata={}, task_id=None)
        nudged = engine._drain_pending_messages([], ctx)
        self.assertEqual(nudged[0]["metadata"]["source"], "task_notification")

    def test_drain_clears_external_queue(self) -> None:
        engine = _make_engine()
        engine.enqueue_external_event("first")
        engine.enqueue_external_event("second")
        ctx = TurnContext(session_id="s", iteration=0, metadata={}, task_id=None)
        engine._drain_pending_messages([], ctx)
        # Second drain returns nothing — queue is empty.
        again = engine._drain_pending_messages([], ctx)
        self.assertEqual(again, [])

    def test_empty_string_ignored(self) -> None:
        engine = _make_engine()
        engine.enqueue_external_event("")
        engine.enqueue_external_event(None)  # type: ignore[arg-type]
        ctx = TurnContext(session_id="s", iteration=0, metadata={}, task_id=None)
        nudged = engine._drain_pending_messages([], ctx)
        self.assertEqual(nudged, [])


class TaskStoreDrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.engine = _make_engine(store=self.store)

    def test_pending_drained_when_context_task_id_set(self) -> None:
        self.store.create(_record("t1"))
        self.store.enqueue_pending_message("t1", "from peer")
        ctx = TurnContext(session_id="s", iteration=0, metadata={}, task_id="t1")
        nudged = self.engine._drain_pending_messages([], ctx)
        self.assertEqual(len(nudged), 1)
        self.assertEqual(nudged[0]["content"], "from peer")
        self.assertEqual(nudged[0]["metadata"]["source"], "send_message")
        # Drain emptied the disk queue.
        self.assertEqual(self.store.drain_pending_messages("t1"), [])

    def test_no_task_id_skips_store_drain(self) -> None:
        self.store.create(_record("t1"))
        self.store.enqueue_pending_message("t1", "would not be visible")
        ctx = TurnContext(session_id="s", iteration=0, metadata={}, task_id=None)
        nudged = self.engine._drain_pending_messages([], ctx)
        self.assertEqual(nudged, [])
        # Message stays queued for whoever IS task t1.
        self.assertEqual(self.store.drain_pending_messages("t1"), ["would not be visible"])

    def test_combines_store_and_external_queue_in_order(self) -> None:
        self.store.create(_record("t1"))
        self.store.enqueue_pending_message("t1", "from-store")
        self.engine.enqueue_external_event("from-external")
        ctx = TurnContext(session_id="s", iteration=0, metadata={}, task_id="t1")
        nudged = self.engine._drain_pending_messages([], ctx)
        self.assertEqual(
            [m["content"] for m in nudged], ["from-store", "from-external"]
        )

    def test_drain_preserves_existing_messages(self) -> None:
        self.store.create(_record("t1"))
        self.store.enqueue_pending_message("t1", "appended")
        existing = [{"role": "user", "content": "original"}]
        ctx = TurnContext(session_id="s", iteration=0, metadata={}, task_id="t1")
        nudged = self.engine._drain_pending_messages(existing, ctx)
        self.assertEqual(len(nudged), 2)
        self.assertEqual(nudged[0]["content"], "original")
        self.assertEqual(nudged[1]["content"], "appended")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
