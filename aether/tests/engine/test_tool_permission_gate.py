from __future__ import annotations

import unittest

from aether import AgentEngine
from aether.agents.middlewares.base import EngineMiddleware
from aether.config.schema import EngineConfig
from aether.models.provider.scripted import ScriptedProvider
from aether.runtime.core.contracts import (
    EngineRequest,
    ExitReason,
    NormalizedResponse,
    ToolCall,
    ToolResult,
    TurnContext,
)
from aether.runtime.session.session_state import SessionMode, clear_mode, set_mode
from aether.runtime.tools.tool_permissions import (
    ToolPermissionDecision,
    ToolPermissionDecisionType,
    ToolPermissionPreview,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.registry import ToolRegistry


class _PermissionPrompter:
    def __init__(self, decisions: list[ToolPermissionDecision]) -> None:
        self.decisions = decisions
        self.requests = []

    def is_interactive(self) -> bool:
        return True

    def request_tool_permission(self, request, *, timeout=None):
        self.requests.append(request)
        if self.decisions:
            return self.decisions.pop(0)
        return ToolPermissionDecision(type=ToolPermissionDecisionType.DENY)


class _WriteSpyTool(ToolExecutor):
    def __init__(self) -> None:
        self.execute_calls: list[ToolCall] = []

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(name="write_file")

    def build_permission_preview(self, call: ToolCall, context: TurnContext):
        return ToolPermissionPreview(
            title="Write file",
            path=str(call.arguments.get("path") or "/tmp/a.txt"),
        )

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        self.execute_calls.append(call)
        return ToolResult(tool_call_id=call.id, name=call.name, content="wrote")


class _AfterToolSpy(EngineMiddleware):
    def __init__(self) -> None:
        self.results: list[ToolResult] = []

    def after_tool(self, result: ToolResult, context: TurnContext) -> ToolResult:
        self.results.append(result)
        return result


class ToolPermissionGateTests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_mode("perm-plan")
        clear_mode("perm-plan-file")

    def _engine_for(self, provider: ScriptedProvider, tool: _WriteSpyTool) -> AgentEngine:
        registry = ToolRegistry()
        registry.register(tool)
        return AgentEngine(
            provider,
            tool_registry=registry,
            config=EngineConfig(max_iterations=5, tool_permissions_enabled=True),
        )

    def test_permission_allow_once_dispatches_tool(self) -> None:
        tool = _WriteSpyTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": "/tmp/a.txt", "content": "x"},
                        )
                    ]
                ),
                NormalizedResponse(content="done"),
            ]
        )
        engine = self._engine_for(provider, tool)
        prompter = _PermissionPrompter(
            [ToolPermissionDecision(type=ToolPermissionDecisionType.ALLOW_ONCE)]
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="perm-allow",
                user_message="write",
                tool_permission_prompter=prompter,
            )
        )

        self.assertEqual(result.final_response, "done")
        self.assertEqual(len(tool.execute_calls), 1)
        self.assertEqual(len(prompter.requests), 1)
        self.assertEqual(result.metadata["tool_permissions"]["allowed_once"], 1)

    def test_permission_reject_does_not_dispatch_tool(self) -> None:
        tool = _WriteSpyTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": "/tmp/a.txt", "content": "x"},
                        )
                    ]
                ),
                NormalizedResponse(content="noted"),
            ]
        )
        engine = self._engine_for(provider, tool)
        after_spy = _AfterToolSpy()
        engine.services.middleware_pipeline.add(after_spy)
        prompter = _PermissionPrompter(
            [ToolPermissionDecision(type=ToolPermissionDecisionType.DENY)]
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="perm-deny",
                user_message="write",
                tool_permission_prompter=prompter,
            )
        )

        self.assertEqual(result.final_response, "noted")
        self.assertEqual(tool.execute_calls, [])
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        self.assertTrue(tool_messages[0]["metadata"]["permission_denied"])
        self.assertFalse(tool_messages[0]["metadata"]["tool_executed"])
        self.assertTrue(after_spy.results[0].metadata["permission_denied"])
        self.assertEqual(result.metadata["tool_permissions"]["denied"], 1)

    def test_permission_abort_stops_turn_without_model_followup(self) -> None:
        tool = _WriteSpyTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": "/tmp/a.txt", "content": "x"},
                        )
                    ]
                ),
            ]
        )
        engine = self._engine_for(provider, tool)
        prompter = _PermissionPrompter(
            [ToolPermissionDecision(type=ToolPermissionDecisionType.ABORT)]
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="perm-abort",
                user_message="write",
                tool_permission_prompter=prompter,
            )
        )

        self.assertEqual(result.exit_reason, ExitReason.INTERRUPTED)
        self.assertIsNone(result.final_response)
        self.assertEqual(tool.execute_calls, [])
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(
            tool_messages[0]["metadata"]["permission_decision"],
            ToolPermissionDecisionType.ABORT.value,
        )
        self.assertEqual(result.metadata["tool_permissions"]["aborted"], 1)

    def test_accept_session_skips_second_prompt_for_same_path(self) -> None:
        tool = _WriteSpyTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": "/tmp/a.txt", "content": "x"},
                        )
                    ]
                ),
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="c2",
                            name="write_file",
                            arguments={"path": "/tmp/a.txt", "content": "y"},
                        )
                    ]
                ),
                NormalizedResponse(content="done"),
            ]
        )
        engine = self._engine_for(provider, tool)
        prompter = _PermissionPrompter(
            [ToolPermissionDecision(type=ToolPermissionDecisionType.ALLOW_SESSION)]
        )

        result = engine.run_turn(
            EngineRequest(
                session_id="perm-session",
                user_message="write twice",
                tool_permission_prompter=prompter,
            )
        )

        self.assertEqual(result.final_response, "done")
        self.assertEqual(len(tool.execute_calls), 2)
        self.assertEqual(len(prompter.requests), 1)
        self.assertEqual(result.metadata["tool_permissions"]["session_rules_added"], 1)
        self.assertEqual(result.metadata["tool_permissions"]["allowed_session"], 2)

    def test_non_interactive_dangerous_tool_denied_by_default(self) -> None:
        tool = _WriteSpyTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": "/tmp/a.txt", "content": "x"},
                        )
                    ]
                ),
                NormalizedResponse(content="noted"),
            ]
        )
        engine = self._engine_for(provider, tool)

        result = engine.run_turn(
            EngineRequest(session_id="perm-noninteractive", user_message="write")
        )

        self.assertEqual(tool.execute_calls, [])
        self.assertEqual(result.final_response, "noted")
        self.assertEqual(result.metadata["tool_permissions"]["non_interactive_denied"], 1)

    def test_plan_mode_blocks_before_permission_prompt(self) -> None:
        tool = _WriteSpyTool()
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": "/tmp/a.txt", "content": "x"},
                        )
                    ]
                ),
                NormalizedResponse(content="blocked"),
            ]
        )
        engine = self._engine_for(provider, tool)
        prompter = _PermissionPrompter(
            [ToolPermissionDecision(type=ToolPermissionDecisionType.ALLOW_ONCE)]
        )
        set_mode("perm-plan", SessionMode.PLAN)

        result = engine.run_turn(
            EngineRequest(
                session_id="perm-plan",
                user_message="write",
                tool_permission_prompter=prompter,
            )
        )

        self.assertEqual(tool.execute_calls, [])
        self.assertEqual(prompter.requests, [])
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        self.assertTrue(tool_messages[0]["metadata"]["plan_mode_blocked"])

    def test_plan_mode_plan_file_write_bypasses_permission_prompt(self) -> None:
        from aether.runtime.session.plan_artifact import get_plan_path

        tool = _WriteSpyTool()
        plan_path = str(get_plan_path("perm-plan-file"))
        provider = ScriptedProvider(
            [
                NormalizedResponse(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="write_file",
                            arguments={"path": plan_path, "content": "x"},
                        )
                    ]
                ),
                NormalizedResponse(content="allowed"),
            ]
        )
        engine = self._engine_for(provider, tool)
        prompter = _PermissionPrompter(
            [ToolPermissionDecision(type=ToolPermissionDecisionType.DENY)]
        )
        set_mode("perm-plan-file", SessionMode.PLAN)

        result = engine.run_turn(
            EngineRequest(
                session_id="perm-plan-file",
                user_message="write plan",
                tool_permission_prompter=prompter,
            )
        )

        self.assertEqual(len(tool.execute_calls), 1)
        self.assertEqual(prompter.requests, [])
        tool_messages = [m for m in result.messages if m.get("role") == "tool"]
        self.assertTrue(tool_messages[0]["metadata"]["plan_mode_plan_file_write"])
        self.assertEqual(result.final_response, "allowed")


class PlanModeGateRefusalTests(unittest.TestCase):
    """Direct unit tests for ``_check_plan_mode_block``.

    Tests the refusal text and the read-only-subagent carve-out
    without spinning up the whole engine loop.
    """

    def setUp(self) -> None:
        self.session_id = "gate-refusal-test"
        set_mode(self.session_id, SessionMode.PLAN)
        self.addCleanup(clear_mode, self.session_id)
        self._context = TurnContext(session_id=self.session_id, iteration=1)

    def test_shell_refusal_lists_read_only_tools(self) -> None:
        # The model commonly regresses to ``shell read_file <path>``
        # or ``shell pip install`` in plan mode (see error2/tool.png).
        # The refusal text must redirect explicitly to the dedicated
        # read tools and call out the "not shell binaries" gotcha.
        from aether.tools.registry import _check_plan_mode_block

        refusal = _check_plan_mode_block(
            "shell", self._context, {"command": "read_file /tmp/x"}
        )
        self.assertIsNotNone(refusal)
        assert refusal is not None  # type checker
        msg = refusal.message
        self.assertIn("'shell' is blocked", msg)
        self.assertIn("read_file", msg)
        self.assertIn("list_dir", msg)
        self.assertIn("grep", msg)
        self.assertIn("glob", msg)
        self.assertIn("not shell binaries", msg)
        self.assertIn("exit_plan_mode", msg)
        self.assertTrue(refusal.metadata["plan_mode_blocked"])

    def test_write_file_refusal_keeps_plan_file_redirect(self) -> None:
        # Sanity check: the generic refusal path (write_file/file_edit/
        # notebook_edit/todo_write/memory_*) still points at the plan
        # file and exit_plan_mode.
        from aether.tools.registry import _check_plan_mode_block

        refusal = _check_plan_mode_block(
            "write_file", self._context, {"path": "/tmp/other.txt", "content": "x"}
        )
        self.assertIsNotNone(refusal)
        assert refusal is not None
        self.assertIn("'write_file' is blocked", refusal.message)
        self.assertIn("plan file", refusal.message)
        self.assertIn("exit_plan_mode", refusal.message)

    def test_task_explore_subagent_allowed_in_plan_mode(self) -> None:
        # The read-only carve-out is the lever that makes the 5-phase
        # workflow's Phase 1 ("launch Explore subagents") actually
        # executable.
        from aether.tools.registry import _check_plan_mode_block

        refusal = _check_plan_mode_block(
            "task",
            self._context,
            {
                "subagent_type": "Explore",
                "description": "find auth helpers",
                "prompt": "...",
            },
        )
        self.assertIsNone(refusal)

    def test_task_plan_subagent_allowed_in_plan_mode(self) -> None:
        from aether.tools.registry import _check_plan_mode_block

        refusal = _check_plan_mode_block(
            "task",
            self._context,
            {"subagent_type": "Plan", "description": "design", "prompt": "..."},
        )
        self.assertIsNone(refusal)

    def test_task_general_purpose_blocked_in_plan_mode(self) -> None:
        # A ``general-purpose`` subagent could trivially mutate state,
        # which would defeat plan-mode's purpose.  Blocked.
        from aether.tools.registry import _check_plan_mode_block

        refusal = _check_plan_mode_block(
            "task",
            self._context,
            {
                "subagent_type": "general-purpose",
                "description": "do stuff",
                "prompt": "...",
            },
        )
        self.assertIsNotNone(refusal)
        assert refusal is not None
        self.assertIn("'task' is blocked", refusal.message)
        self.assertIn("Explore", refusal.message)
        self.assertIn("Plan", refusal.message)


if __name__ == "__main__":
    unittest.main()
