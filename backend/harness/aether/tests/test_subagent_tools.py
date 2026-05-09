"""Tests for the subagent dispatch family.

Sprint 3.5 / PR-2 (PR 3.5.6).  Covers:

* ``AgentTool`` happy path / depth-limit / unwired / spill.
* ``TaskOutputTool`` v1 stub behaviour.
* ``TaskStopTool`` route-through behaviour.
* ``SubagentManager.stop_task`` + ``_stop_events`` lifecycle.
"""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any

from aether.config.schema import EngineConfig
from aether.runtime.contracts import ToolCall, TurnContext
from aether.subagents.contracts import SubagentResult, SubagentStatus, SubagentTask
from aether.subagents.manager import SubagentManager
from aether.tools.builtins.agent_tool import AgentTool
from aether.tools.builtins.task_output import TaskOutputTool
from aether.tools.builtins.task_stop import TaskStopTool


def _ctx(
    *,
    parent: Any | None = None,
    manager: Any | None = None,
    config: EngineConfig | None = None,
    session_id: str = "ses-A",
) -> TurnContext:
    md: dict[str, Any] = {"_engine_config": config or EngineConfig()}
    if parent is not None:
        md["_parent_agent"] = parent
    if manager is not None:
        md["_subagent_manager"] = manager
    return TurnContext(session_id=session_id, iteration=0, metadata=md)


# -------------------------------------------------------------- AgentTool


class _StubManager:
    def __init__(self, *, result: SubagentResult | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[SubagentTask] = []

    def run_task(self, *, parent, task: SubagentTask) -> SubagentResult:
        self.calls.append(task)
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result

    def stop_task(self, task_id: str) -> bool:
        return False


def _ok_result(task_id: str, summary: str = "all done") -> SubagentResult:
    return SubagentResult(
        task_id=task_id,
        status=SubagentStatus.COMPLETED,
        summary=summary,
        engine_result=None,
        error=None,
        duration_seconds=0.5,
        metadata={"subagent_id": "sub-xyz"},
    )


class AgentToolTests(unittest.TestCase):
    def test_a1_happy_path_returns_summary(self) -> None:
        manager = _StubManager(result=_ok_result("task-x", "great success"))
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool()
        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "do the thing"}),
            _ctx(parent=parent, manager=manager),
        )
        self.assertFalse(result.is_error, result.content)
        self.assertIn("Subagent task complete", result.content)
        self.assertIn("status: COMPLETED", result.content)
        self.assertIn("great success", result.content)
        self.assertEqual(len(manager.calls), 1)
        self.assertIn("do the thing", manager.calls[0].request.user_message)

    def test_a2_depth_limit_runtime_error_surfaced(self) -> None:
        manager = _StubManager(exc=RuntimeError("Delegation depth limit reached"))
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool()
        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "p"}),
            _ctx(parent=parent, manager=manager),
        )
        self.assertTrue(result.is_error)
        self.assertIn("dispatch failed", result.content)
        self.assertIn("Delegation depth", result.content)

    def test_a3_general_exception_surfaced(self) -> None:
        manager = _StubManager(exc=ValueError("boom"))
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool()
        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "p"}),
            _ctx(parent=parent, manager=manager),
        )
        self.assertTrue(result.is_error)
        self.assertIn("crashed", result.content)

    def test_a4_no_parent_returns_unwired_error(self) -> None:
        tool = AgentTool()
        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "p"}),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("not wired", result.content)

    def test_a5_failed_subagent_marks_error(self) -> None:
        manager = _StubManager(
            result=SubagentResult(
                task_id="task-fail",
                status=SubagentStatus.FAILED,
                summary=None,
                engine_result=None,
                error="provider crash",
                duration_seconds=1.0,
                metadata={"subagent_id": "sub-fail"},
            )
        )
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool()
        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "p"}),
            _ctx(parent=parent, manager=manager),
        )
        self.assertTrue(result.is_error)
        self.assertIn("provider crash", result.content)
        self.assertEqual(result.metadata["subagent_status"], "FAILED")

    def test_a6_subagent_dispatch_disabled_via_config(self) -> None:
        manager = _StubManager(result=_ok_result("t"))
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        cfg = EngineConfig(allow_subagent_dispatch=False)
        tool = AgentTool()
        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "p"}),
            _ctx(parent=parent, manager=manager, config=cfg),
        )
        self.assertTrue(result.is_error)
        self.assertIn("disabled", result.content)

    def test_a7_oversize_summary_spills(self) -> None:
        manager = _StubManager(result=_ok_result("t-big", "X" * 80_000))
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool()
        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "p"}),
            _ctx(parent=parent, manager=manager, session_id="ses-spill-task"),
        )
        self.assertFalse(result.is_error)
        self.assertIn("output truncated", result.content)

    def test_a8_expected_output_included_in_format(self) -> None:
        manager = _StubManager(result=_ok_result("t"))
        parent = SimpleNamespace(delegate_depth=0, subagent_manager=manager)
        tool = AgentTool()
        result = tool.execute(
            ToolCall(
                id="c1",
                name="task",
                arguments={"prompt": "p", "expected_output": "list of paths"},
            ),
            _ctx(parent=parent, manager=manager),
        )
        self.assertIn("expected_output: list of paths", result.content)

    def test_a9_empty_prompt_rejected(self) -> None:
        tool = AgentTool()
        result = tool.execute(
            ToolCall(id="c1", name="task", arguments={"prompt": "  "}),
            _ctx(parent=SimpleNamespace(delegate_depth=0, subagent_manager=_StubManager(result=_ok_result("t")))),
        )
        self.assertTrue(result.is_error)


# ----------------------------------------------------------- TaskOutputTool


class TaskOutputTests(unittest.TestCase):
    def test_b1_always_returns_not_supported(self) -> None:
        tool = TaskOutputTool()
        result = tool.execute(
            ToolCall(id="c1", name="task_output", arguments={"task_id": "t-1"}),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("not supported in synchronous", result.content)
        self.assertEqual(result.metadata["task_id"], "t-1")

    def test_b2_missing_task_id_rejected(self) -> None:
        tool = TaskOutputTool()
        result = tool.execute(
            ToolCall(id="c1", name="task_output", arguments={}),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("task_id", result.content)


# ----------------------------------------------------------- TaskStopTool


class _StoppingManager:
    def __init__(self, known: bool = True) -> None:
        self._known = known
        self.calls: list[str] = []

    def stop_task(self, task_id: str) -> bool:
        self.calls.append(task_id)
        return self._known


class TaskStopTests(unittest.TestCase):
    def test_c1_running_task_signal_delivered(self) -> None:
        mgr = _StoppingManager(known=True)
        tool = TaskStopTool(subagent_manager=mgr)
        result = tool.execute(
            ToolCall(id="c1", name="task_stop", arguments={"task_id": "t-1"}),
            _ctx(manager=mgr),
        )
        self.assertFalse(result.is_error)
        self.assertEqual(mgr.calls, ["t-1"])
        self.assertTrue(result.metadata["delivered"])

    def test_c2_unknown_task_returns_error(self) -> None:
        mgr = _StoppingManager(known=False)
        tool = TaskStopTool(subagent_manager=mgr)
        result = tool.execute(
            ToolCall(id="c1", name="task_stop", arguments={"task_id": "missing"}),
            _ctx(manager=mgr),
        )
        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)
        self.assertFalse(result.metadata["delivered"])

    def test_c3_no_manager_returns_unavailable(self) -> None:
        tool = TaskStopTool()
        result = tool.execute(
            ToolCall(id="c1", name="task_stop", arguments={"task_id": "t"}),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("unavailable", result.content)

    def test_c4_disabled_via_config(self) -> None:
        cfg = EngineConfig(allow_subagent_dispatch=False)
        tool = TaskStopTool(subagent_manager=_StoppingManager(known=True))
        result = tool.execute(
            ToolCall(id="c1", name="task_stop", arguments={"task_id": "t"}),
            _ctx(config=cfg),
        )
        self.assertTrue(result.is_error)
        self.assertIn("disabled", result.content)


# ---------------------------------------------------- SubagentManager.stop_task


class _PausingChildAgent:
    """Stand-in for a child AgentEngine used in stop_task tests.

    Blocks on ``proceed_event`` so the parent thread can race a
    ``stop_task`` call against the child's still-running state.  Records
    interrupt reasons so we can assert the manager forwarded them.
    """

    def __init__(self, *, hold_seconds: float = 0.2) -> None:
        self.subagent_id = "child-pause"
        self.hold_seconds = hold_seconds
        self.interrupt_reasons: list[str] = []
        self.proceed_event = threading.Event()

    @property
    def delegate_depth(self) -> int:
        return 1

    def interrupt(self, *, reason: str | None = None) -> None:
        self.interrupt_reasons.append(reason or "")
        self.proceed_event.set()

    def run_loop(self, request):
        # Wait for either the proceed event (set by interrupt) or the
        # max hold period — whichever comes first.
        self.proceed_event.wait(timeout=self.hold_seconds)
        from aether.runtime.contracts import EngineStatus
        return SimpleNamespace(
            status=EngineStatus.INTERRUPTED if self.proceed_event.is_set() else EngineStatus.COMPLETED,
            final_response=None,
            error=None,
        )


class _StubBuilder:
    def __init__(self, child: _PausingChildAgent) -> None:
        self._child = child
        self.built = 0

    def build_child(self, *, parent, task, child_depth):
        self.built += 1
        return self._child


class _ParentForBuilder:
    def __init__(self) -> None:
        self.delegate_depth = 0
        self._registered: list[Any] = []

    def _register_child(self, child) -> None:
        self._registered.append(child)

    def _unregister_child(self, child) -> None:
        if child in self._registered:
            self._registered.remove(child)


class SubagentManagerStopTests(unittest.TestCase):
    def test_d1_stop_task_unknown_returns_false(self) -> None:
        mgr = SubagentManager()
        self.assertFalse(mgr.stop_task("never-existed"))
        self.assertFalse(mgr.stop_task(""))

    def test_d2_stop_task_running_delivers_interrupt(self) -> None:
        child = _PausingChildAgent(hold_seconds=2.0)
        builder = _StubBuilder(child)
        mgr = SubagentManager(builder=builder)
        parent = _ParentForBuilder()
        from aether.runtime.contracts import EngineRequest
        from aether.config.schema import ModelCallConfig
        task = SubagentTask(
            task_id="task-pause",
            goal="g",
            request=EngineRequest(session_id="s", model_config=ModelCallConfig()),
        )

        # Run the manager in a background thread so the test thread
        # can race a stop_task against the child's wait.
        completed = threading.Event()

        def runner() -> None:
            mgr.run_task(parent=parent, task=task)
            completed.set()

        t = threading.Thread(target=runner)
        t.start()
        # Give the manager a moment to register the child.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not mgr.is_task_running("task-pause"):
            time.sleep(0.005)
        self.assertTrue(mgr.is_task_running("task-pause"), "child never registered")
        ok = mgr.stop_task("task-pause")
        self.assertTrue(ok)
        completed.wait(timeout=2.0)
        t.join(timeout=1.0)
        self.assertGreaterEqual(len(child.interrupt_reasons), 1)
        self.assertFalse(mgr.is_task_running("task-pause"))

    def test_d3_stop_event_cleaned_up_after_completion(self) -> None:
        child = _PausingChildAgent(hold_seconds=0.05)  # finishes quickly
        builder = _StubBuilder(child)
        mgr = SubagentManager(builder=builder)
        parent = _ParentForBuilder()
        from aether.runtime.contracts import EngineRequest
        from aether.config.schema import ModelCallConfig
        task = SubagentTask(
            task_id="task-quick",
            goal="g",
            request=EngineRequest(session_id="s", model_config=ModelCallConfig()),
        )
        mgr.run_task(parent=parent, task=task)
        self.assertFalse(mgr.is_task_running("task-quick"))
        # stop_task on already-finished task returns False instead of raising.
        self.assertFalse(mgr.stop_task("task-quick"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
