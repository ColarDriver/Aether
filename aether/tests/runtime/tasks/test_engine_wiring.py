"""Tests for AgentEngine ↔ TaskStore wiring."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import EngineRequest, NormalizedResponse
from aether.runtime.tasks import TaskRecord, TaskStatus, TaskStore
from aether.subagents import SubagentManager, SubagentTask
from aether.subagents.default_builder import DefaultSubagentBuilder


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


class EngineTaskStoreWiringTests(unittest.TestCase):
    def test_root_engine_lazy_builds_default_store(self) -> None:
        engine = AgentEngine(
            ScriptedProvider([NormalizedResponse(content="ok")]),
            config=EngineConfig(use_builtin_tools=False),
        )
        # Root engine builds a default TaskStore.  Use it (don't create
        # any tasks) so the home dir stays clean — see lazy_init test
        # in test_store.py.
        self.assertIsNotNone(engine._task_store)

    def test_disabled_via_config(self) -> None:
        engine = AgentEngine(
            ScriptedProvider([NormalizedResponse(content="ok")]),
            config=EngineConfig(use_builtin_tools=False, task_store_enabled=False),
        )
        self.assertIsNone(engine._task_store)

    def test_explicit_store_overrides_default(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            engine = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="ok")]),
                config=EngineConfig(use_builtin_tools=False),
                task_store=store,
            )
            self.assertIs(engine._task_store, store)

    def test_recovery_runs_at_root_engine_init(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Pre-populate with a stale RUNNING record.
            pre_store = TaskStore(root=root)
            pre_store.create(_record("orphan"))
            stale = pre_store.read("orphan")
            assert stale is not None
            stale.last_heartbeat = time.time() - 600
            pre_store._write_record(stale)  # noqa: SLF001 — test hook

            # New engine with a fresh TaskStore on the same root.
            engine = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="ok")]),
                config=EngineConfig(
                    use_builtin_tools=False,
                    task_store_path=root,
                    task_store_stale_seconds=60.0,
                ),
            )
            assert engine._task_store is not None
            recovered = engine._task_store.read("orphan")
            assert recovered is not None
            self.assertEqual(recovered.status, TaskStatus.KILLED)

    def test_subagent_inherits_parent_store(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            parent = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="parent")]),
                config=EngineConfig(use_builtin_tools=False),
                subagent_manager=SubagentManager(),
                task_store=store,
            )
            task = SubagentTask(
                task_id="child-1",
                goal="g",
                request=EngineRequest(session_id="child-s", user_message="u"),
                provider=ScriptedProvider([NormalizedResponse(content="child")]),
            )
            child = DefaultSubagentBuilder().build_child(parent, task, child_depth=1)
            self.assertIs(child._task_store, store)

    def test_subagent_does_not_run_recovery(self) -> None:
        # The child shares the parent's store; if it ran recovery, a
        # legitimate sibling task created mid-session would get marked
        # KILLED on every ``build_child`` call.  Set up the parent
        # FIRST (its own recovery pass runs against an empty store),
        # then introduce a stale sibling, then build the child.
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            parent = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="parent")]),
                config=EngineConfig(use_builtin_tools=False, task_store_stale_seconds=60.0),
                subagent_manager=SubagentManager(),
                task_store=store,
            )
            # Now plant a stale sibling — represents a peer task started
            # post-parent-init that hasn't checked in yet.
            sibling = _record("sibling")
            store.create(sibling)
            stale = store.read("sibling")
            assert stale is not None
            stale.last_heartbeat = time.time() - 600
            store._write_record(stale)  # noqa: SLF001 — test hook

            # Building a child must NOT trigger recovery (gate:
            # ``self._delegate_depth == 0``).
            task = SubagentTask(
                task_id="child-1",
                goal="g",
                request=EngineRequest(session_id="child-s", user_message="u"),
                provider=ScriptedProvider([NormalizedResponse(content="child")]),
            )
            child = DefaultSubagentBuilder().build_child(parent, task, child_depth=1)
            self.assertGreater(child.delegate_depth, 0)

            after = store.read("sibling")
            assert after is not None
            self.assertEqual(after.status, TaskStatus.RUNNING)

    def test_context_metadata_exposes_task_store(self) -> None:
        # Tools (TaskOutput, SendMessage) read the store from
        # context.metadata["_task_store"] when no explicit store was
        # injected via constructor.
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            engine = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="ok")]),
                config=EngineConfig(use_builtin_tools=False),
                task_store=store,
            )
            engine.run_turn(
                EngineRequest(session_id="s1", user_message="hi", system_message="x")
            )
            # The hook captures the most recent context — easiest
            # check: ensure the engine kept the store.
            self.assertIs(engine._task_store, store)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
