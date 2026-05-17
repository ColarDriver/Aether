"""Tests that ExitPlanModeTool persists the plan artifact (PR 12.4)."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from aether.runtime.core.contracts import ToolCall, TurnContext
from aether.runtime.session import clear_mode, set_mode
from aether.runtime.session.plan_artifact import get_plan_path, read_plan
from aether.runtime.session.session_state import SessionMode
from aether.tools.builtins.exit_plan_mode import ExitPlanModeTool


class _StubApprovingPrompter:
    def __init__(self, approve: bool) -> None:
        self._approve = approve
        self.calls: list[str] = []

    def is_interactive(self) -> bool:
        return True

    def confirm_plan(self, plan: str, *, context) -> bool:  # noqa: ARG002
        self.calls.append(plan)
        return self._approve

    def ask_questions(self, questions, *, timeout=None):  # noqa: ARG002
        raise NotImplementedError


class _ExitPlanModeCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = mock.patch.dict(
            os.environ, {"AETHER_HOME": self._tmp.name}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        self.session_id = "exit-plan-test"
        set_mode(self.session_id, SessionMode.PLAN)
        self.addCleanup(clear_mode, self.session_id)

    def _ctx(self, *, prompter) -> TurnContext:
        return TurnContext(
            session_id=self.session_id,
            iteration=0,
            metadata={"_approval_prompter": prompter},
        )

    def _execute(self, *, approve: bool, plan: str = "# Plan\n\n- one\n- two"):
        prompter = _StubApprovingPrompter(approve)
        tool = ExitPlanModeTool()
        result = tool.execute(
            ToolCall(id="c1", name="exit_plan_mode", arguments={"plan": plan}),
            self._ctx(prompter=prompter),
        )
        return result, prompter


class ApproveWritesArtifactTests(_ExitPlanModeCase):
    def test_artifact_written_before_approval(self) -> None:
        _, prompter = self._execute(approve=True)
        # Prompter saw the plan and approved — artifact must be on disk.
        self.assertEqual(len(prompter.calls), 1)
        self.assertEqual(read_plan(self.session_id), "# Plan\n\n- one\n- two")

    def test_artifact_path_reported_in_metadata(self) -> None:
        result, _ = self._execute(approve=True)
        self.assertTrue(result.metadata.get("approved"))
        self.assertEqual(
            result.metadata.get("plan_path"),
            str(get_plan_path(self.session_id)),
        )


class RejectKeepsArtifactTests(_ExitPlanModeCase):
    def test_reject_keeps_artifact_on_disk(self) -> None:
        result, _ = self._execute(approve=False, plan="# v1\n\n- todo")
        self.assertFalse(result.metadata.get("approved"))
        # Plan stays available so the model (or user via /plan) can
        # revise.
        self.assertEqual(read_plan(self.session_id), "# v1\n\n- todo")

    def test_reject_then_resubmit_overwrites_artifact(self) -> None:
        self._execute(approve=False, plan="# v1\n\n- step a")
        self._execute(approve=False, plan="# v2\n\n- step a\n- step b")
        self.assertEqual(read_plan(self.session_id), "# v2\n\n- step a\n- step b")

    def test_plan_path_in_reject_metadata(self) -> None:
        result, _ = self._execute(approve=False)
        self.assertEqual(
            result.metadata.get("plan_path"),
            str(get_plan_path(self.session_id)),
        )


class PlanCurrentReadsArtifactTests(unittest.TestCase):
    """Round-trip via the gateway RPC: write plan, plan.current reads it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = mock.patch.dict(
            os.environ, {"AETHER_HOME": self._tmp.name}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        from aether.cli.sessions import SessionRecord, save_session
        from aether.gateway.dispatcher import (
            register_builtins,
            reset_dispatcher_for_tests,
        )
        from aether.gateway.handlers import register_handler_methods
        from aether.gateway.handlers.state import reset_state_for_tests

        reset_dispatcher_for_tests()
        reset_state_for_tests()
        register_builtins()
        register_handler_methods()
        self.session_id = "plan-current-test"
        save_session(
            SessionRecord.new(
                session_id=self.session_id,
                provider="openai",
                model="gpt-4",
            )
        )

    def _ok(self, name: str, params: dict | None = None) -> dict:
        from aether.gateway.dispatcher import dispatch_request
        from aether.gateway.protocol import RpcRequest, RpcResponse

        resp = dispatch_request(RpcRequest(id="x", method=name, params=params))
        assert isinstance(resp, RpcResponse)
        if resp.error is not None:
            self.fail(f"{name} unexpected error: {resp.error.message}")
        assert resp.result is not None
        return resp.result

    def test_plan_current_returns_artifact_after_write(self) -> None:
        from aether.runtime.session.plan_artifact import write_plan

        write_plan(self.session_id, "# Strategy\n\n- step one")
        result = self._ok("plan.current", {"session_id": self.session_id})
        self.assertTrue(result["has_plan"])
        self.assertEqual(result["plan_content"], "# Strategy\n\n- step one")
        self.assertIsNotNone(result["plan_path"])
        self.assertTrue(result["plan_path"].endswith(".md"))

    def test_plan_current_returns_no_plan_when_artifact_missing(self) -> None:
        result = self._ok("plan.current", {"session_id": self.session_id})
        self.assertFalse(result["has_plan"])
        self.assertIsNone(result["plan_content"])
        self.assertIsNone(result["plan_path"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
