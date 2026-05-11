from __future__ import annotations

import unittest
from typing import Any

from aether import AgentEngine
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.contracts import (
    EngineRequest,
    EngineStatus,
    ExitReason,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.hooks import EngineHooks
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _ResourceTool(ToolExecutor):
    def __init__(
        self,
        name: str = "resource",
        *,
        release_log: list[str] | None = None,
        raise_on_execute: bool = False,
        raise_on_release: bool = False,
        interrupt_on_execute: bool = False,
    ) -> None:
        self._name = name
        self.raise_on_execute = raise_on_execute
        self.raise_on_release = raise_on_release
        self.interrupt_on_execute = interrupt_on_execute
        self.acquire_calls: list[str] = []
        self.execute_calls: list[str] = []
        self.release_calls: list[str] = []
        self.release_log = release_log

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name=self._name)

    def acquire_task_resource(self, task_id: str) -> None:
        self.acquire_calls.append(task_id)

    def release_task_resource(self, task_id: str) -> None:
        self.release_calls.append(task_id)
        if self.release_log is not None:
            self.release_log.append(self._name)
        if self.raise_on_release:
            raise RuntimeError(f"{self._name} release failed")

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        self.execute_calls.append(call.id)
        if self.interrupt_on_execute:
            parent = context.metadata.get("_parent_agent")
            if parent is not None:
                parent.interrupt(context.session_id, "stop during tool")
        if self.raise_on_execute:
            raise RuntimeError(f"{self._name} execute failed")
        return ToolResult(tool_call_id=call.id, name=call.name, content=f"{self._name}:ok")


class _CleanupHooks(EngineHooks):
    def __init__(self, *, raise_on_cleanup: bool = False) -> None:
        self.raise_on_cleanup = raise_on_cleanup
        self.cleanup_calls: list[dict[str, Any]] = []
        self.session_end_calls: list[tuple[bool, bool]] = []

    def on_task_cleanup(
        self,
        *,
        task_id: str,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        self.cleanup_calls.append(
            {
                "task_id": task_id,
                "session_id": session_id,
                "completed": completed,
                "interrupted": interrupted,
                "context_task_id": context_metadata.get("task_id"),
            }
        )
        if self.raise_on_cleanup:
            raise RuntimeError("cleanup hook failed")

    def on_session_end(
        self,
        *,
        session_id: str,
        completed: bool,
        interrupted: bool,
        context_metadata: dict[str, Any],
    ) -> None:
        self.session_end_calls.append((completed, interrupted))


def _registry_with(*tools: ToolExecutor) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


class TaskResourceCleanupTests(unittest.TestCase):
    def test_success_releases_each_task_resource_once_in_reverse_order(self) -> None:
        release_log: list[str] = []
        first = _ResourceTool("first", release_log=release_log)
        second = _ResourceTool("second", release_log=release_log)
        hooks = _CleanupHooks()
        engine = AgentEngine(
            ScriptedProvider(
                [
                    NormalizedResponse(
                        content="",
                        tool_calls=[
                            ToolCall(id="call-1", name="first", arguments={"n": 1}),
                            ToolCall(id="call-2", name="first", arguments={"n": 2}),
                            ToolCall(id="call-3", name="second"),
                        ],
                    ),
                    NormalizedResponse(content="done"),
                ]
            ),
            tool_registry=_registry_with(first, second),
            hooks=hooks,
            config=EngineConfig(max_iterations=4, use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="cleanup-success",
                user_message="go",
                metadata={"task_id": "task-success"},
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(first.execute_calls, ["call-1", "call-2"])
        self.assertEqual(first.acquire_calls, ["task-success"])
        self.assertEqual(second.acquire_calls, ["task-success"])
        self.assertEqual(first.release_calls, ["task-success"])
        self.assertEqual(second.release_calls, ["task-success"])
        self.assertEqual(release_log, ["second", "first"])
        self.assertEqual(
            result.metadata["resource_cleanup"],
            {
                "completed": True,
                "interrupted": False,
                "acquired": 2,
                "released": 2,
                "errors": [],
                "hook_called": True,
            },
        )
        self.assertEqual(
            result.metadata["turn"]["resource_cleanup"],
            result.metadata["resource_cleanup"],
        )
        self.assertNotIn("_task_resource_handles", result.metadata["turn"])
        self.assertEqual(
            hooks.cleanup_calls,
            [
                {
                    "task_id": "task-success",
                    "session_id": "cleanup-success",
                    "completed": True,
                    "interrupted": False,
                    "context_task_id": "task-success",
                }
            ],
        )
        self.assertEqual(hooks.session_end_calls, [(True, False)])

    def test_provider_failure_after_tool_still_releases_resource(self) -> None:
        tool = _ResourceTool()
        engine = AgentEngine(
            ScriptedProvider(
                [
                    NormalizedResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-1", name="resource")],
                    )
                ]
            ),
            tool_registry=_registry_with(tool),
            config=EngineConfig(max_iterations=4, use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="cleanup-provider-failure",
                user_message="go",
                metadata={"task_id": "task-provider-failure"},
            )
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.PROVIDER_ERROR)
        self.assertEqual(tool.release_calls, ["task-provider-failure"])
        self.assertEqual(result.metadata["resource_cleanup"]["completed"], False)
        self.assertEqual(result.metadata["resource_cleanup"]["interrupted"], False)
        self.assertEqual(result.metadata["resource_cleanup"]["released"], 1)

    def test_tool_failure_still_releases_resource_without_overwriting_error(self) -> None:
        tool = _ResourceTool(raise_on_execute=True)
        engine = AgentEngine(
            ScriptedProvider(
                [
                    NormalizedResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-1", name="resource")],
                    )
                ]
            ),
            tool_registry=_registry_with(tool),
            config=EngineConfig(
                fail_on_tool_error=True,
                max_iterations=4,
                use_builtin_tools=False,
            ),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="cleanup-tool-failure",
                user_message="go",
                metadata={"task_id": "task-tool-failure"},
            )
        )

        self.assertEqual(result.status, EngineStatus.FAILED)
        self.assertEqual(result.exit_reason, ExitReason.TOOL_ERROR)
        self.assertIn("execute failed", result.error or "")
        self.assertEqual(tool.release_calls, ["task-tool-failure"])
        self.assertEqual(result.metadata["resource_cleanup"]["released"], 1)
        self.assertEqual(result.metadata["resource_cleanup"]["errors"], [])

    def test_interrupt_after_tool_execution_releases_and_marks_interrupted(self) -> None:
        tool = _ResourceTool(interrupt_on_execute=True)
        hooks = _CleanupHooks()
        engine = AgentEngine(
            ScriptedProvider(
                [
                    NormalizedResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-1", name="resource")],
                    )
                ]
            ),
            tool_registry=_registry_with(tool),
            hooks=hooks,
            config=EngineConfig(max_iterations=4, use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="cleanup-interrupt",
                user_message="go",
                metadata={"task_id": "task-interrupt"},
            )
        )

        self.assertEqual(result.status, EngineStatus.INTERRUPTED)
        self.assertEqual(result.exit_reason, ExitReason.INTERRUPTED)
        self.assertEqual(tool.release_calls, ["task-interrupt"])
        self.assertEqual(
            result.metadata["resource_cleanup"],
            {
                "completed": False,
                "interrupted": True,
                "acquired": 1,
                "released": 1,
                "errors": [],
                "hook_called": True,
            },
        )
        self.assertEqual(hooks.cleanup_calls[0]["interrupted"], True)
        self.assertEqual(hooks.session_end_calls, [(False, True)])

    def test_release_failure_is_recorded_without_overwriting_success(self) -> None:
        tool = _ResourceTool(raise_on_release=True)
        engine = AgentEngine(
            ScriptedProvider(
                [
                    NormalizedResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-1", name="resource")],
                    ),
                    NormalizedResponse(content="done"),
                ]
            ),
            tool_registry=_registry_with(tool),
            config=EngineConfig(max_iterations=4, use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="cleanup-release-failure",
                user_message="go",
                metadata={"task_id": "task-release-failure"},
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(result.final_response, "done")
        cleanup = result.metadata["resource_cleanup"]
        self.assertEqual(cleanup["acquired"], 1)
        self.assertEqual(cleanup["released"], 0)
        self.assertEqual(len(cleanup["errors"]), 1)
        self.assertEqual(cleanup["errors"][0]["resource"], "resource")
        self.assertIn("release failed", cleanup["errors"][0]["error"])

    def test_cleanup_hook_failure_is_recorded_and_session_end_still_runs(self) -> None:
        tool = _ResourceTool()
        hooks = _CleanupHooks(raise_on_cleanup=True)
        engine = AgentEngine(
            ScriptedProvider(
                [
                    NormalizedResponse(
                        content="",
                        tool_calls=[ToolCall(id="call-1", name="resource")],
                    ),
                    NormalizedResponse(content="done"),
                ]
            ),
            tool_registry=_registry_with(tool),
            hooks=hooks,
            config=EngineConfig(max_iterations=4, use_builtin_tools=False),
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="cleanup-hook-failure",
                user_message="go",
                metadata={"task_id": "task-hook-failure"},
            )
        )

        self.assertEqual(result.status, EngineStatus.COMPLETED)
        self.assertEqual(tool.release_calls, ["task-hook-failure"])
        self.assertEqual(hooks.session_end_calls, [(True, False)])
        cleanup = result.metadata["resource_cleanup"]
        self.assertNotIn("hook_called", cleanup)
        self.assertEqual(len(cleanup["errors"]), 1)
        self.assertEqual(cleanup["errors"][0]["resource"], "hook:on_task_cleanup")
        self.assertIn("cleanup hook failed", cleanup["errors"][0]["error"])

    def test_cleanup_skips_release_and_hook_when_task_id_missing(self) -> None:
        tool = _ResourceTool()
        hooks = _CleanupHooks()
        engine = AgentEngine(
            ScriptedProvider([NormalizedResponse(content="unused")]),
            hooks=hooks,
            config=EngineConfig(use_builtin_tools=False),
        )
        context = TurnContext(
            session_id="cleanup-no-task",
            iteration=1,
            metadata={"_task_resource_handles": [tool]},
            task_id=None,
        )

        cleanup = engine._cleanup_task_resources(
            context=context,
            completed=True,
            interrupted=False,
        )

        self.assertEqual(cleanup["completed"], True)
        self.assertEqual(cleanup["interrupted"], False)
        self.assertEqual(cleanup["released"], 0)
        self.assertEqual(tool.release_calls, [])
        self.assertEqual(hooks.cleanup_calls, [])
        self.assertTrue(context.metadata["_task_cleanup_done"])


if __name__ == "__main__":
    unittest.main()
