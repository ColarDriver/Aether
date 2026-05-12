"""Tests for the plan-mode + ask-user-question interaction tools.

Sprint 3.5 / PR-2 (PR 3.5.7).  Covers:

* ``EnterPlanModeTool`` switching session mode + subagent guard.
* ``ExitPlanModeTool`` calling the prompter and toggling mode.
* ``AskUserQuestionTool`` validation, prompter contract, subagent
  guard, non-interactive fallback.
* ``ToolRegistry.dispatch`` blocking write-class tools while the
  session is in plan mode.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Mapping

from aether.cli.approval_prompter import StubPrompter
from aether.config.schema import EngineConfig
from aether.runtime.core.contracts import ToolCall, ToolResult, TurnContext
from aether.runtime.session.session_state import (
    SessionMode,
    all_sessions,
    clear_mode,
    get_mode,
    set_mode,
)
from aether.tools.base import ToolDescriptor, ToolExecutor
from aether.tools.builtins.ask_user_question import AskUserQuestionTool
from aether.tools.builtins.enter_plan_mode import EnterPlanModeTool
from aether.tools.builtins.exit_plan_mode import ExitPlanModeTool
from aether.tools.registry import ToolRegistry


def _ctx(
    *,
    session_id: str = "ses-plan",
    config: EngineConfig | None = None,
    parent_agent: Any | None = None,
    prompter: Any | None = None,
) -> TurnContext:
    md: dict[str, Any] = {"_engine_config": config or EngineConfig()}
    if parent_agent is not None:
        md["_parent_agent"] = parent_agent
    if prompter is not None:
        md["_approval_prompter"] = prompter
    return TurnContext(session_id=session_id, iteration=0, metadata=md)


def _reset_all_modes() -> None:
    for sid in list(all_sessions().keys()):
        clear_mode(sid)


# ---------------------------------------------------------- enter_plan_mode


class EnterPlanModeTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all_modes()

    def tearDown(self) -> None:
        _reset_all_modes()

    def test_b1_flips_mode_to_plan(self) -> None:
        tool = EnterPlanModeTool()
        result = tool.execute(
            ToolCall(id="c1", name="enter_plan_mode", arguments={}),
            _ctx(session_id="ses1"),
        )
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(get_mode("ses1"), "plan")
        self.assertIn("enter_plan_mode", "enter_plan_mode")  # stable
        self.assertIn("read_file", result.content)
        self.assertEqual(result.metadata["new_mode"], "plan")

    def test_b2_subagent_context_rejected(self) -> None:
        tool = EnterPlanModeTool()
        parent = SimpleNamespace(delegate_depth=1)
        result = tool.execute(
            ToolCall(id="c1", name="enter_plan_mode", arguments={}),
            _ctx(session_id="ses-sub", parent_agent=parent),
        )
        self.assertTrue(result.is_error)
        self.assertIn("subagent", result.content)
        self.assertEqual(get_mode("ses-sub"), "agent")

    def test_b3_disabled_via_config(self) -> None:
        cfg = EngineConfig(plan_mode_enabled=False)
        tool = EnterPlanModeTool()
        result = tool.execute(
            ToolCall(id="c1", name="enter_plan_mode", arguments={}),
            _ctx(session_id="ses-off", config=cfg),
        )
        self.assertTrue(result.is_error)
        self.assertIn("disabled", result.content)

    def test_b4_includes_full_guidance(self) -> None:
        tool = EnterPlanModeTool()
        result = tool.execute(
            ToolCall(id="c1", name="enter_plan_mode", arguments={}),
            _ctx(session_id="sesguid"),
        )
        self.assertIn("read_file", result.content)
        self.assertIn("exit_plan_mode", result.content)


# ----------------------------------------------------------- exit_plan_mode


class ExitPlanModeTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all_modes()

    def tearDown(self) -> None:
        _reset_all_modes()

    def _enter_plan(self, sid: str) -> None:
        set_mode(sid, SessionMode.PLAN)

    def test_d1_approved_returns_to_agent_mode(self) -> None:
        sid = "ses-d1"
        self._enter_plan(sid)
        prompter = StubPrompter(approve_plan=True)
        tool = ExitPlanModeTool(prompter=prompter)
        result = tool.execute(
            ToolCall(id="c1", name="exit_plan_mode", arguments={"plan": "step 1\nstep 2"}),
            _ctx(session_id=sid),
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["approved"], True)
        self.assertEqual(get_mode(sid), "agent")
        self.assertEqual(prompter.confirm_calls, ["step 1\nstep 2"])

    def test_d2_rejected_keeps_plan_mode(self) -> None:
        sid = "ses-d2"
        self._enter_plan(sid)
        prompter = StubPrompter(approve_plan=False)
        tool = ExitPlanModeTool(prompter=prompter)
        result = tool.execute(
            ToolCall(id="c1", name="exit_plan_mode", arguments={"plan": "p"}),
            _ctx(session_id=sid),
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.metadata["approved"], False)
        self.assertEqual(get_mode(sid), "plan")

    def test_d3_called_outside_plan_mode_rejected(self) -> None:
        tool = ExitPlanModeTool(prompter=StubPrompter())
        result = tool.execute(
            ToolCall(id="c1", name="exit_plan_mode", arguments={"plan": "p"}),
            _ctx(session_id="sesfresh"),
        )
        self.assertTrue(result.is_error)
        self.assertIn("not in plan mode", result.content)

    def test_d4_no_prompter_returns_error_keeps_plan_mode(self) -> None:
        sid = "ses-d4"
        self._enter_plan(sid)
        tool = ExitPlanModeTool()
        result = tool.execute(
            ToolCall(id="c1", name="exit_plan_mode", arguments={"plan": "p"}),
            _ctx(session_id=sid),
        )
        self.assertTrue(result.is_error)
        self.assertEqual(get_mode(sid), "plan")
        self.assertIn("plan_preview", result.metadata)

    def test_d5_empty_plan_rejected(self) -> None:
        tool = ExitPlanModeTool(prompter=StubPrompter())
        result = tool.execute(
            ToolCall(id="c1", name="exit_plan_mode", arguments={"plan": "   "}),
            _ctx(),
        )
        self.assertTrue(result.is_error)

    def test_d6_prompter_picked_from_metadata(self) -> None:
        sid = "ses-d6"
        self._enter_plan(sid)
        prompter = StubPrompter(approve_plan=True)
        tool = ExitPlanModeTool()
        result = tool.execute(
            ToolCall(id="c1", name="exit_plan_mode", arguments={"plan": "p"}),
            _ctx(session_id=sid, prompter=prompter),
        )
        self.assertFalse(result.is_error)
        self.assertEqual(get_mode(sid), "agent")
        self.assertEqual(prompter.confirm_calls, ["p"])


# ----------------------------------------------------------- write-tool gate


class _NoopTool(ToolExecutor):
    """Records dispatch hits so we can prove the gate fires *before* execute."""

    def __init__(self, name: str) -> None:
        self._descriptor = ToolDescriptor(name=name, description="x", parameters={"type": "object"}, required=[])
        self.executed = 0

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def execute(self, call: ToolCall, context: TurnContext) -> ToolResult:
        self.executed += 1
        return ToolResult(tool_call_id=call.id, name=call.name, content="ran", is_error=False)


class PlanModeRegistryGateTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all_modes()

    def tearDown(self) -> None:
        _reset_all_modes()

    def _registry(self, name: str) -> tuple[ToolRegistry, _NoopTool]:
        tool = _NoopTool(name)
        reg = ToolRegistry()
        reg.register(tool)
        return reg, tool

    def test_c1_plan_mode_blocks_shell(self) -> None:
        reg, t = self._registry("shell")
        set_mode("ses-block", SessionMode.PLAN)
        result = reg.dispatch(
            ToolCall(id="c1", name="shell", arguments={"command": "ls"}),
            _ctx(session_id="ses-block"),
        )
        self.assertTrue(result.is_error)
        self.assertEqual(t.executed, 0)
        self.assertTrue(result.metadata.get("plan_mode_blocked"))

    def test_c2_plan_mode_blocks_write_file(self) -> None:
        reg, t = self._registry("write_file")
        set_mode("sw", SessionMode.PLAN)
        result = reg.dispatch(
            ToolCall(id="c1", name="write_file", arguments={"path": "x", "content": "y"}),
            _ctx(session_id="sw"),
        )
        self.assertTrue(result.is_error)
        self.assertEqual(t.executed, 0)

    def test_c3_plan_mode_blocks_file_edit(self) -> None:
        reg, t = self._registry("file_edit")
        set_mode("se", SessionMode.PLAN)
        result = reg.dispatch(
            ToolCall(id="c1", name="file_edit", arguments={}),
            _ctx(session_id="se"),
        )
        self.assertTrue(result.is_error)
        self.assertEqual(t.executed, 0)

    def test_c4_plan_mode_blocks_notebook_edit(self) -> None:
        reg, t = self._registry("notebook_edit")
        set_mode("sn", SessionMode.PLAN)
        result = reg.dispatch(
            ToolCall(id="c1", name="notebook_edit", arguments={}),
            _ctx(session_id="sn"),
        )
        self.assertTrue(result.is_error)
        self.assertEqual(t.executed, 0)

    def test_c4b_plan_mode_blocks_todo_write(self) -> None:
        reg, t = self._registry("todo_write")
        set_mode("st", SessionMode.PLAN)
        result = reg.dispatch(
            ToolCall(id="c1", name="todo_write", arguments={"todos": []}),
            _ctx(session_id="st"),
        )
        self.assertTrue(result.is_error)
        self.assertEqual(t.executed, 0)

    def test_c4c_plan_mode_blocks_subagent_dispatch(self) -> None:
        reg, t = self._registry("task")
        set_mode("st2", SessionMode.PLAN)
        result = reg.dispatch(
            ToolCall(id="c1", name="task", arguments={"prompt": "x"}),
            _ctx(session_id="st2"),
        )
        self.assertTrue(result.is_error)
        self.assertEqual(t.executed, 0)

    def test_c5_plan_mode_allows_read_file(self) -> None:
        reg, t = self._registry("read_file")
        set_mode("sr", SessionMode.PLAN)
        result = reg.dispatch(
            ToolCall(id="c1", name="read_file", arguments={"path": "x"}),
            _ctx(session_id="sr"),
        )
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(t.executed, 1)

    def test_c6_plan_mode_allows_grep(self) -> None:
        reg, t = self._registry("grep")
        set_mode("sg", SessionMode.PLAN)
        result = reg.dispatch(
            ToolCall(id="c1", name="grep", arguments={}),
            _ctx(session_id="sg"),
        )
        self.assertFalse(result.is_error)

    def test_c7_agent_mode_allows_shell(self) -> None:
        reg, t = self._registry("shell")
        # Default: agent mode.
        result = reg.dispatch(
            ToolCall(id="c1", name="shell", arguments={"command": "ls"}),
            _ctx(session_id="sa"),
        )
        self.assertFalse(result.is_error)
        self.assertEqual(t.executed, 1)


# ---------------------------------------------------------- ask_user_question


class _MappingPrompter:
    def __init__(self, *, answers: Mapping[str, Any], interactive: bool = True, raise_exc: Exception | None = None) -> None:
        self._answers = dict(answers)
        self._interactive = interactive
        self._raise = raise_exc
        self.calls: list[Any] = []

    def is_interactive(self) -> bool:
        return self._interactive

    def confirm_plan(self, plan: str, *, context: Any | None = None) -> bool:
        return True

    def ask_questions(self, questions, *, timeout=None):
        self.calls.append(list(questions))
        if self._raise is not None:
            raise self._raise
        return self._answers


class AskUserQuestionTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_all_modes()

    def tearDown(self) -> None:
        _reset_all_modes()

    def test_e1_basic_response_formatting(self) -> None:
        p = _MappingPrompter(answers={"q1": "yes"})
        tool = AskUserQuestionTool(prompter=p)
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={"questions": [{"id": "q1", "prompt": "Continue?"}]},
            ),
            _ctx(),
        )
        self.assertFalse(result.is_error, result.content)
        self.assertIn("User responses", result.content)
        self.assertIn("Continue?", result.content)
        self.assertIn("yes", result.content)
        self.assertEqual(result.metadata["question_count"], 1)
        self.assertEqual(result.metadata["answer_count"], 1)

    def test_e2_subagent_context_rejected(self) -> None:
        parent = SimpleNamespace(delegate_depth=2)
        tool = AskUserQuestionTool(prompter=_MappingPrompter(answers={}))
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={"questions": [{"id": "q1", "prompt": "p"}]},
            ),
            _ctx(parent_agent=parent),
        )
        self.assertTrue(result.is_error)
        self.assertIn("subagent", result.content)

    def test_e3_non_interactive_returns_error(self) -> None:
        p = _MappingPrompter(answers={}, interactive=False)
        tool = AskUserQuestionTool(prompter=p)
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={"questions": [{"id": "q1", "prompt": "p"}]},
            ),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("non-interactive", result.content)

    def test_e4_timeout_surfaces_seconds(self) -> None:
        p = _MappingPrompter(answers={}, raise_exc=TimeoutError("t"))
        cfg = EngineConfig(ask_user_question_timeout_seconds=42)
        tool = AskUserQuestionTool(prompter=p)
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={"questions": [{"id": "q1", "prompt": "p"}]},
            ),
            _ctx(config=cfg),
        )
        self.assertTrue(result.is_error)
        self.assertIn("42", result.content)

    def test_e5_options_round_trip(self) -> None:
        p = _MappingPrompter(answers={"q1": "alpha"})
        tool = AskUserQuestionTool(prompter=p)
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={
                    "questions": [
                        {
                            "id": "q1",
                            "prompt": "Pick one",
                            "options": [
                                {"id": "alpha", "label": "Alpha"},
                                {"id": "beta", "label": "Beta"},
                            ],
                        }
                    ]
                },
            ),
            _ctx(),
        )
        self.assertFalse(result.is_error)
        self.assertIn("alpha", result.content)

    def test_e6_multi_select_renders_each_choice(self) -> None:
        p = _MappingPrompter(answers={"q1": ["a", "b"]})
        tool = AskUserQuestionTool(prompter=p)
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={
                    "questions": [
                        {
                            "id": "q1",
                            "prompt": "Pick many",
                            "allow_multiple": True,
                            "options": [
                                {"id": "a", "label": "A"},
                                {"id": "b", "label": "B"},
                            ],
                        }
                    ]
                },
            ),
            _ctx(),
        )
        self.assertFalse(result.is_error)
        self.assertIn("- a", result.content)
        self.assertIn("- b", result.content)

    def test_e8_multiple_questions(self) -> None:
        p = _MappingPrompter(answers={"q1": "first", "q2": "second"})
        tool = AskUserQuestionTool(prompter=p)
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={
                    "questions": [
                        {"id": "q1", "prompt": "P1"},
                        {"id": "q2", "prompt": "P2"},
                    ]
                },
            ),
            _ctx(),
        )
        self.assertFalse(result.is_error)
        self.assertIn("first", result.content)
        self.assertIn("second", result.content)
        self.assertEqual(result.metadata["answer_count"], 2)

    def test_e9_disabled_via_config(self) -> None:
        cfg = EngineConfig(ask_user_question_enabled=False)
        tool = AskUserQuestionTool(prompter=_MappingPrompter(answers={}))
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={"questions": [{"id": "q1", "prompt": "p"}]},
            ),
            _ctx(config=cfg),
        )
        self.assertTrue(result.is_error)
        self.assertIn("disabled", result.content)

    def test_e10_validation_empty_question_id_rejected(self) -> None:
        tool = AskUserQuestionTool(prompter=_MappingPrompter(answers={}))
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={"questions": [{"id": "", "prompt": "p"}]},
            ),
            _ctx(),
        )
        self.assertTrue(result.is_error)

    def test_e11_validation_no_questions_rejected(self) -> None:
        tool = AskUserQuestionTool(prompter=_MappingPrompter(answers={}))
        result = tool.execute(
            ToolCall(id="c1", name="ask_user_question", arguments={"questions": []}),
            _ctx(),
        )
        self.assertTrue(result.is_error)

    def test_e12_no_prompter_returns_error(self) -> None:
        tool = AskUserQuestionTool()
        result = tool.execute(
            ToolCall(
                id="c1",
                name="ask_user_question",
                arguments={"questions": [{"id": "q1", "prompt": "p"}]},
            ),
            _ctx(),
        )
        self.assertTrue(result.is_error)
        self.assertIn("no approval prompter", result.content.lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
