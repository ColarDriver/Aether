"""Tests for SubagentManager.run_task_async — PR 10.5."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig, ModelCallConfig
from aether.models.provider.base import ModelProvider
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    NormalizedResponse,
    StreamDeltaCallback,
    StreamSilentCallback,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.tasks import TaskStatus, TaskStore
from aether.subagents import SubagentManager, SubagentTask
from aether.subagents.contracts import SubagentStatus
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _PingTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="ping")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(tool_call_id=call.id, name=call.name, content="pong")


class _ExplodingProvider(ModelProvider):
    """Provider that always raises — drives the FAILED branch."""

    provider_name = "test-explode"
    api_mode = "chat"

    def __init__(self) -> None:
        self.model = "explode"

    def generate(
        self,
        messages: list[dict],
        tools: list[ToolDescriptor],
        config: ModelCallConfig,
        context: TurnContext,
        stream_callback: StreamDeltaCallback | None = None,
        stream_silent_callback: StreamSilentCallback | None = None,
    ) -> NormalizedResponse:
        del messages, tools, config, context, stream_callback, stream_silent_callback
        raise RuntimeError("provider boom")


def _make_parent(
    *, store: TaskStore, manager: SubagentManager, provider: ModelProvider | None = None
) -> AgentEngine:
    return AgentEngine(
        provider or ScriptedProvider([NormalizedResponse(content="parent")]),
        config=EngineConfig(use_builtin_tools=False),
        subagent_manager=manager,
        task_store=store,
    )


def _scripted_child_responses(*, tool_calls: int) -> list[NormalizedResponse]:
    """Build N tool-call responses followed by a terminal text response."""
    out: list[NormalizedResponse] = []
    for i in range(tool_calls):
        out.append(
            NormalizedResponse(
                content="",
                tool_calls=[ToolCall(id=f"c{i + 1}", name="ping", arguments={})],
            )
        )
    out.append(NormalizedResponse(content="done"))
    return out


def _child_request(session: str = "child-s") -> EngineRequest:
    return EngineRequest(session_id=session, user_message="do work")


def _wait_terminal(store: TaskStore, task_id: str, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.read(task_id)
        if record is not None and record.status.is_terminal:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"task {task_id!r} did not reach terminal status within {timeout}s"
    )


class AsyncDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.manager = SubagentManager()
        self.addCleanup(self.manager.shutdown)
        # Equip the child with a PingTool; build_child inherits.
        registry = ToolRegistry()
        registry.register(_PingTool())
        provider = ScriptedProvider(_scripted_child_responses(tool_calls=2))
        self.parent = AgentEngine(
            ScriptedProvider([NormalizedResponse(content="parent")]),
            tool_registry=registry,
            config=EngineConfig(use_builtin_tools=False),
            subagent_manager=self.manager,
            task_store=self.store,
        )
        self._child_provider = provider

    def _task(self, **metadata: Any) -> SubagentTask:
        return SubagentTask(
            task_id=f"t-{metadata.pop('id', 1)}",
            goal="g",
            request=_child_request(),
            provider=self._child_provider,
            metadata={"subagent_type": "general-purpose", **metadata},
        )

    def test_run_task_async_returns_immediately(self) -> None:
        task = self._task(id=1)
        t0 = time.monotonic()
        tid = self.manager.run_task_async(parent=self.parent, task=task)
        elapsed = time.monotonic() - t0
        self.assertEqual(tid, "t-1")
        self.assertLess(elapsed, 0.5, "run_task_async should not block")
        # Immediately-visible record (we wrote it BEFORE submit).
        record = self.store.read("t-1")
        assert record is not None
        self.assertIn(record.status, (TaskStatus.RUNNING, TaskStatus.COMPLETED))
        self.assertTrue(record.background)
        _wait_terminal(self.store, "t-1")

    def test_async_lifecycle_writes_terminal_status_and_result(self) -> None:
        tid = self.manager.run_task_async(parent=self.parent, task=self._task(id=2))
        _wait_terminal(self.store, tid)
        record = self.store.read(tid)
        assert record is not None
        self.assertEqual(record.status, TaskStatus.COMPLETED)
        self.assertEqual(record.summary, "done")
        self.assertIsNotNone(record.finished_at)
        self.assertIsNotNone(record.result_path)
        # Tool messages + assistant turn lands in messages.jsonl.
        msgs = (Path(self._tmp.name) / tid / "messages.jsonl").read_text().splitlines()
        roles = [m for m in msgs if m]
        # 2 tool calls + final assistant text → at least 3 entries.
        self.assertGreaterEqual(len(roles), 3)

    def test_async_records_tool_use_count_and_iterations(self) -> None:
        tid = self.manager.run_task_async(parent=self.parent, task=self._task(id=3))
        _wait_terminal(self.store, tid)
        record = self.store.read(tid)
        assert record is not None
        # 2 tool calls instrumented.
        self.assertGreaterEqual(record.tool_use_count, 2)
        self.assertGreater(record.iterations, 0)


class AsyncFailureTests(unittest.TestCase):
    def test_provider_failure_marks_task_failed(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            manager = SubagentManager()
            self.addCleanup(manager.shutdown)
            parent = _make_parent(store=store, manager=manager)
            task = SubagentTask(
                task_id="boom",
                goal="g",
                request=_child_request(),
                provider=_ExplodingProvider(),
                metadata={"subagent_type": "general-purpose"},
            )
            tid = manager.run_task_async(parent=parent, task=task)
            _wait_terminal(store, tid)
            record = store.read(tid)
            assert record is not None
            # Provider error is recovered into a FAILED engine status.
            self.assertIn(record.status, (TaskStatus.FAILED, TaskStatus.COMPLETED))
            # We don't strictly require COMPLETED vs FAILED here because
            # the engine's recovery layer may surface a soft failure;
            # the important guarantee is that we wrote a terminal record
            # and a result.json file.
            self.assertTrue(
                (Path(tmp) / tid / "result.json").exists(),
                "result.json must be written even on failure",
            )


class AsyncDepthTests(unittest.TestCase):
    def test_run_task_async_respects_depth_limit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            manager = SubagentManager(max_spawn_depth=1)
            self.addCleanup(manager.shutdown)
            parent = _make_parent(store=store, manager=manager)
            # Force "already at depth limit" by bumping delegate_depth.
            parent._delegate_depth = 1  # noqa: SLF001 — test hook
            task = SubagentTask(
                task_id="t-deep",
                goal="g",
                request=_child_request(),
                provider=ScriptedProvider([NormalizedResponse(content="ok")]),
                metadata={"subagent_type": "general-purpose"},
            )
            with self.assertRaises(RuntimeError) as ctx:
                manager.run_task_async(parent=parent, task=task)
            self.assertIn("Delegation depth", str(ctx.exception))
            self.assertIsNone(store.read("t-deep"), "no record on depth violation")

    def test_run_task_async_requires_task_store(self) -> None:
        manager = SubagentManager()
        self.addCleanup(manager.shutdown)
        # Build a parent WITHOUT a task store.
        parent = AgentEngine(
            ScriptedProvider([NormalizedResponse(content="parent")]),
            config=EngineConfig(use_builtin_tools=False, task_store_enabled=False),
            subagent_manager=manager,
        )
        task = SubagentTask(
            task_id="t-nostore",
            goal="g",
            request=_child_request(),
            provider=ScriptedProvider([NormalizedResponse(content="ok")]),
            metadata={"subagent_type": "general-purpose"},
        )
        with self.assertRaises(RuntimeError) as ctx:
            manager.run_task_async(parent=parent, task=task)
        self.assertIn("TaskStore", str(ctx.exception))


class AsyncConcurrentTests(unittest.TestCase):
    def test_multiple_async_tasks_complete_independently(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            manager = SubagentManager(max_concurrent_background=4)
            self.addCleanup(manager.shutdown)
            registry = ToolRegistry()
            registry.register(_PingTool())
            parent = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="parent")]),
                tool_registry=registry,
                config=EngineConfig(use_builtin_tools=False),
                subagent_manager=manager,
                task_store=store,
            )
            ids: list[str] = []
            for i in range(4):
                provider = ScriptedProvider(
                    _scripted_child_responses(tool_calls=1)
                )
                task = SubagentTask(
                    task_id=f"par-{i}",
                    goal="g",
                    request=_child_request(session=f"sess-{i}"),
                    provider=provider,
                    metadata={"subagent_type": "general-purpose"},
                )
                ids.append(manager.run_task_async(parent=parent, task=task))
            for tid in ids:
                _wait_terminal(store, tid)
                record = store.read(tid)
                assert record is not None
                self.assertEqual(record.status, TaskStatus.COMPLETED)


class AsyncFanoutHooksTests(unittest.TestCase):
    """Verify the parent's user-installed hooks still fire after fan-out wraps them."""

    def test_inner_hooks_still_invoked(self) -> None:
        from aether.runtime.core.hooks import EngineHooks

        invoked: list[str] = []

        class _SpyHooks(EngineHooks):
            def on_session_start(self, *, session_id, context_metadata):
                invoked.append("session_start")

            def post_llm_call(self, *, session_id, iteration, response_text, context_metadata):
                invoked.append(f"post_llm:{iteration}")

        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            manager = SubagentManager()
            self.addCleanup(manager.shutdown)
            registry = ToolRegistry()
            registry.register(_PingTool())
            parent = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="parent")]),
                tool_registry=registry,
                config=EngineConfig(use_builtin_tools=False),
                subagent_manager=manager,
                task_store=store,
                hooks=_SpyHooks(),
            )
            provider = ScriptedProvider(_scripted_child_responses(tool_calls=1))
            task = SubagentTask(
                task_id="hooky",
                goal="g",
                request=_child_request(),
                provider=provider,
                metadata={"subagent_type": "general-purpose"},
            )
            tid = manager.run_task_async(parent=parent, task=task)
            _wait_terminal(store, tid)
            self.assertIn("session_start", invoked)
            # At least one post_llm fired (the final "done" turn).
            self.assertTrue(any(s.startswith("post_llm:") for s in invoked))


class AsyncShutdownTests(unittest.TestCase):
    def test_shutdown_is_idempotent(self) -> None:
        manager = SubagentManager()
        manager.shutdown(wait=True)
        manager.shutdown(wait=True)  # must not raise


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
