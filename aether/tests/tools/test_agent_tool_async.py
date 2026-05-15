"""Tests for AgentTool's run_in_background branch — PR 10.5."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aether import AgentEngine
from aether.agents.types import AgentTypeDefinition, AgentTypeRegistry
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.tasks import TaskStatus, TaskStore
from aether.subagents import SubagentManager
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.builtins.agent_tool import AgentTool
from aether.tools.registry import ToolRegistry


class _PingTool(ToolExecutor):
    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="ping")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        del context
        return ToolResult(tool_call_id=call.id, name=call.name, content="pong")


def _make_parent(
    *,
    store: TaskStore,
    manager: SubagentManager,
    provider: ScriptedProvider | None = None,
    config: EngineConfig | None = None,
) -> AgentEngine:
    """Build a parent engine with a provider whose script the **child**
    inherits via ``DefaultSubagentBuilder``.

    These tests never call ``parent.run_turn``, so the script is shaped
    for what the child consumes.
    """
    registry = ToolRegistry()
    registry.register(_PingTool())
    return AgentEngine(
        provider or ScriptedProvider([NormalizedResponse(content="child-done")]),
        tool_registry=registry,
        config=config or EngineConfig(use_builtin_tools=False),
        subagent_manager=manager,
        task_store=store,
    )


def _wait_terminal(store: TaskStore, task_id: str, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = store.read(task_id)
        if record is not None and record.status.is_terminal:
            return
        time.sleep(0.02)
    raise AssertionError(f"task {task_id!r} not terminal within {timeout}s")


class AgentToolDescriptorTests(unittest.TestCase):
    def test_descriptor_exposes_run_in_background_default_false(self) -> None:
        tool = AgentTool()
        props = tool.descriptor.parameters["properties"]
        self.assertIn("run_in_background", props)
        self.assertEqual(props["run_in_background"]["type"], "boolean")
        self.assertFalse(props["run_in_background"]["default"])


class AgentToolAsyncBranchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = TaskStore(root=Path(self._tmp.name))
        self.manager = SubagentManager()
        self.addCleanup(self.manager.shutdown)
        # ScriptedProvider for the child — provider override is set via
        # the subagent task in the manager-level tests, but here we go
        # through AgentTool.execute which doesn't take a child provider
        # parameter.  Instead we rely on the parent's provider for the
        # child (DefaultSubagentBuilder inherits parent's provider).
        # Parent's provider is what the child inherits via the builder.
        self.parent = _make_parent(
            store=self.store,
            manager=self.manager,
            provider=ScriptedProvider([NormalizedResponse(content="child-done")]),
        )

    def _execute(self, args: dict, *, registry: AgentTypeRegistry | None = None) -> ToolResult:
        registry = registry or AgentTypeRegistry(search_paths=[])
        tool = AgentTool(
            parent_agent=self.parent,
            subagent_manager=self.manager,
            agent_type_registry=registry,
        )
        ctx = TurnContext(session_id="parent-s", iteration=0, metadata={})
        return tool.execute(
            ToolCall(id="c1", name="task", arguments=args), ctx
        )

    def test_run_in_background_returns_async_launched_metadata(self) -> None:
        result = self._execute(
            {
                "subagent_type": "general-purpose",
                "prompt": "run in background",
                "run_in_background": True,
            }
        )
        self.assertFalse(result.is_error, msg=result.content)
        self.assertEqual(result.metadata.get("status"), "async_launched")
        tid = result.metadata.get("task_id")
        assert isinstance(tid, str) and tid.startswith("task-")
        self.assertEqual(result.metadata.get("subagent_type"), "general-purpose")
        self.assertTrue(result.metadata.get("background"))
        # output_path / result_path point to the task store.
        self.assertIn(tid, str(result.metadata.get("output_path")))
        # Wait for the background task to wrap up so the test cleans up
        # cleanly (TemporaryDirectory cleanup must not race writes).
        _wait_terminal(self.store, tid)

    def test_async_branch_does_not_block(self) -> None:
        t0 = time.monotonic()
        result = self._execute(
            {
                "subagent_type": "general-purpose",
                "prompt": "run in background",
                "run_in_background": True,
            }
        )
        elapsed = time.monotonic() - t0
        self.assertFalse(result.is_error)
        self.assertLess(elapsed, 0.5)
        _wait_terminal(self.store, result.metadata["task_id"])

    def test_invalid_run_in_background_type_returns_error(self) -> None:
        result = self._execute(
            {
                "subagent_type": "general-purpose",
                "prompt": "go",
                "run_in_background": "yes",  # not a bool
            }
        )
        self.assertTrue(result.is_error)
        self.assertIn("run_in_background", result.content)

    def test_async_disabled_via_config_returns_error(self) -> None:
        # Re-build parent with async disabled.
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            manager = SubagentManager()
            self.addCleanup(manager.shutdown)
            registry = ToolRegistry()
            registry.register(_PingTool())
            parent = AgentEngine(
                ScriptedProvider([NormalizedResponse(content="ok")]),
                tool_registry=registry,
                config=EngineConfig(
                    use_builtin_tools=False,
                    subagent_async_enabled=False,
                ),
                subagent_manager=manager,
                task_store=store,
            )
            tool = AgentTool(
                parent_agent=parent,
                subagent_manager=manager,
                agent_type_registry=AgentTypeRegistry(search_paths=[]),
            )
            ctx = TurnContext(session_id="s", iteration=0, metadata={})
            # The engine's _prepare_turn_entry normally injects
            # _engine_config; here we wire it manually since we bypass
            # run_loop and call execute directly.
            ctx.metadata["_engine_config"] = parent.config
            result = tool.execute(
                ToolCall(
                    id="c1",
                    name="task",
                    arguments={"prompt": "go", "run_in_background": True},
                ),
                ctx,
            )
            self.assertTrue(result.is_error)
            self.assertIn("disabled", result.content.lower())


class TypeBackgroundForcesAsyncTests(unittest.TestCase):
    def test_type_definition_background_true_forces_async_path(self) -> None:
        with TemporaryDirectory() as tmp:
            store = TaskStore(root=Path(tmp))
            manager = SubagentManager()
            self.addCleanup(manager.shutdown)
            parent = _make_parent(
                store=store,
                manager=manager,
                provider=ScriptedProvider([NormalizedResponse(content="child-done")]),
            )
            # Custom type with background=True; AgentTool must auto-async.
            forced = AgentTypeDefinition(
                agent_type="forced-async",
                description="bg only",
                background=True,
            )
            registry = AgentTypeRegistry(search_paths=[])
            registry.discover()
            registry._types[forced.agent_type] = forced  # noqa: SLF001 — test injection

            tool = AgentTool(
                parent_agent=parent,
                subagent_manager=manager,
                agent_type_registry=registry,
            )
            ctx = TurnContext(session_id="s", iteration=0, metadata={})
            # Caller did NOT pass run_in_background.
            result = tool.execute(
                ToolCall(
                    id="c1",
                    name="task",
                    arguments={
                        "subagent_type": "forced-async",
                        "prompt": "go",
                    },
                ),
                ctx,
            )
            self.assertFalse(result.is_error, msg=result.content)
            self.assertEqual(result.metadata.get("status"), "async_launched")
            _wait_terminal(store, result.metadata["task_id"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
