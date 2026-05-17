"""Tests for the <task-notification> routing — PR 10.7."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.subagents import SubagentManager, SubagentTask
from aether.subagents.contracts import SubagentResult, SubagentStatus
from aether.subagents.manager import _build_task_notification, _xml_escape
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.builtins.agent_tool import AgentTool
from aether.tools.registry import ToolRegistry


# ---------------------------------------------------------------- XML format

class NotificationXmlTests(unittest.TestCase):
    def test_includes_required_fields(self) -> None:
        result = SubagentResult(
            task_id="t-1",
            status=SubagentStatus.COMPLETED,
            summary="found 5 matches",
            engine_result=None,
            duration_seconds=2.34,
            metadata={"subagent_type": "Explore"},
        )
        xml = _build_task_notification("t-1", result)
        self.assertIn("<task-notification>", xml)
        self.assertIn("</task-notification>", xml)
        self.assertIn("<task_id>t-1</task_id>", xml)
        self.assertIn("<subagent_type>Explore</subagent_type>", xml)
        self.assertIn("<status>completed</status>", xml)
        self.assertIn("<duration_seconds>2.3</duration_seconds>", xml)
        self.assertIn("<summary>found 5 matches</summary>", xml)

    def test_escapes_special_chars(self) -> None:
        result = SubagentResult(
            task_id="t-1",
            status=SubagentStatus.FAILED,
            summary="found & in <foo> tag",
            engine_result=None,
            error="tag <bar> & noise",
            duration_seconds=0.1,
            metadata={"subagent_type": "general-purpose"},
        )
        xml = _build_task_notification("t-1", result)
        self.assertIn("&amp;", xml)
        self.assertIn("&lt;foo&gt;", xml)
        self.assertIn("&lt;bar&gt;", xml)
        self.assertNotIn("<foo>", xml)

    def test_summary_and_error_optional(self) -> None:
        result = SubagentResult(
            task_id="t-1",
            status=SubagentStatus.INTERRUPTED,
            summary=None,
            engine_result=None,
            error=None,
            duration_seconds=0.0,
            metadata={"subagent_type": "Plan"},
        )
        xml = _build_task_notification("t-1", result)
        self.assertNotIn("<summary>", xml)
        self.assertNotIn("<error>", xml)
        self.assertIn("<status>interrupted</status>", xml)

    def test_xml_escape_helper(self) -> None:
        self.assertEqual(_xml_escape("a & b"), "a &amp; b")
        self.assertEqual(_xml_escape("<x>"), "&lt;x&gt;")


# ---------------------------------------------------------------- routing

class _PingTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="ping")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(tool_call_id=call.id, name=call.name, content="pong")


def _wait_terminal(store: TaskStore, task_id: str, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.read(task_id)
        if record is not None and record.status.is_terminal:
            return
        time.sleep(0.02)
    raise AssertionError(f"task {task_id!r} did not finish within {timeout}s")


class RootEngineNotificationTests(unittest.TestCase):
    """End-to-end: spawn an async child via AgentTool, verify the parent
    root engine sees the <task-notification> in its external queue."""

    def test_async_child_completion_routes_to_root_engine_queue(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            manager = SubagentManager()
            self.addCleanup(manager.shutdown)

            registry = ToolRegistry()
            registry.register(_PingTool())
            parent = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="child-done")]),
                tool_registry=registry,
                config=EngineConfig(use_builtin_tools=False),
                subagent_manager=manager,
                task_store=store,
            )
            # Pretend the parent already opened a session — needed so
            # the manager's notification routing can find this engine
            # by parent_session_id.
            parent._current_session_id = "parent-s"  # noqa: SLF001 - test seam
            manager.register_root_engine(parent)

            tool = AgentTool(
                parent_agent=parent,
                subagent_manager=manager,
            )
            ctx = TurnContext(session_id="parent-s", iteration=0, metadata={})
            result = tool.execute(
                ToolCall(
                    id="c1",
                    name="task",
                    arguments={
                        "prompt": "do it",
                        "run_in_background": True,
                    },
                ),
                ctx,
            )
            self.assertFalse(result.is_error, msg=result.content)
            tid = result.metadata["task_id"]
            _wait_terminal(store, tid)

            # Brief moment for the done-callback to dispatch the
            # notification (executor runs the callback off-thread).
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not parent._external_event_queue:
                time.sleep(0.01)
            self.assertEqual(len(parent._external_event_queue), 1)
            notification = parent._external_event_queue[0]
            self.assertIn("<task-notification>", notification)
            self.assertIn(tid, notification)
            self.assertIn("<status>completed</status>", notification)


class NestedNotificationTests(unittest.TestCase):
    """When the parent of an async child is itself an async task,
    the notification should land in the parent's TaskStore pending
    queue rather than an in-memory engine queue."""

    def test_nested_async_routes_via_task_store(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            manager = SubagentManager()
            self.addCleanup(manager.shutdown)

            registry = ToolRegistry()
            registry.register(_PingTool())
            # Root engine — needed so the manager can resolve the
            # store to look up parent_task_id.
            root = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="ok")]),
                tool_registry=registry,
                config=EngineConfig(use_builtin_tools=False),
                subagent_manager=manager,
                task_store=store,
            )
            root._current_session_id = "root-s"  # noqa: SLF001
            manager.register_root_engine(root)

            # Pretend the "parent" is itself an async task already in
            # the store (e.g. previously spawned with
            # run_in_background=True).
            parent_task_id = "task-parent"
            now = time.time()
            store.create(
                TaskRecord(
                    task_id=parent_task_id,
                    parent_session_id="root-s",
                    subagent_type="general-purpose",
                    prompt="parent work",
                    status=TaskStatus.RUNNING,
                    started_at=now,
                    last_heartbeat=now,
                    background=True,
                )
            )
            # Build a SubagentResult that claims the parent is
            # ``task-parent`` and feed it directly through the
            # manager's dispatcher — easier than racing a real
            # nested async lifecycle.
            child_result = SubagentResult(
                task_id="task-child",
                status=SubagentStatus.COMPLETED,
                summary="child done",
                engine_result=None,
                duration_seconds=1.5,
                metadata={
                    "subagent_type": "Explore",
                    "parent_task_id": parent_task_id,
                    "parent_session_id": "root-s",
                },
            )
            manager._dispatch_completion_notification("task-child", child_result)

            # Notification is in the parent's pending queue, not the
            # root engine's external queue.
            queued = store.drain_pending_messages(parent_task_id)
            self.assertEqual(len(queued), 1)
            self.assertIn("<task-notification>", queued[0])
            self.assertIn("task-child", queued[0])
            self.assertEqual(root._external_event_queue, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
