"""Tests for ``plan.*`` gateway methods — Sprint 12 PR 12.1."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from aether.cli.sessions import SessionRecord, load_session, save_session
from aether.gateway.dispatcher import (
    dispatch_request,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.handlers.state import reset_state_for_tests, set_current_session
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    RpcRequest,
    RpcResponse,
)
from aether.runtime.session import all_sessions, clear_mode, set_mode
from aether.runtime.session.plan_artifact import get_plan_path, read_plan, write_plan
from aether.runtime.session.session_state import SessionMode


class _PlanMethodsCase(unittest.TestCase):
    """Per-test AETHER_HOME isolation + clean dispatcher / session state."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = mock.patch.dict(
            os.environ, {"AETHER_HOME": self._tmp.name}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        reset_dispatcher_for_tests()
        reset_state_for_tests()
        register_builtins()
        register_handler_methods()

        # Clear any process-level session_state leakage from prior tests.
        for sid in list(all_sessions()):
            clear_mode(sid)

    def _make_session(self, session_id: str = "s-test") -> str:
        record = SessionRecord.new(
            session_id=session_id,
            provider="openai",
            model="gpt-4",
        )
        save_session(record)
        return session_id

    def _ok(self, name: str, params: dict | None = None) -> dict:
        resp = dispatch_request(RpcRequest(id="x", method=name, params=params))
        assert isinstance(resp, RpcResponse)
        if resp.error is not None:
            self.fail(f"{name} unexpected error: {resp.error.code} {resp.error.message}")
        assert resp.result is not None
        return resp.result

    def _err(self, name: str, params: dict | None = None):
        resp = dispatch_request(RpcRequest(id="x", method=name, params=params))
        assert isinstance(resp, RpcResponse)
        if resp.error is None:
            self.fail(f"{name} should have errored; got {resp.result!r}")
        return resp.error


# ---------------------------------------------------------------- catalog

class CatalogTests(_PlanMethodsCase):
    def test_plan_is_present_in_catalog_with_session_category(self) -> None:
        result = self._ok("commands.catalog")
        by_name = {entry["name"]: entry for entry in result["commands"]}
        self.assertIn("/plan", by_name)
        self.assertEqual(
            by_name["/plan"]["description"],
            "Enable plan mode or view the current session plan",
        )
        self.assertEqual(by_name["/plan"]["category"], "session")


# ---------------------------------------------------------------- mode_get

class ModeGetTests(_PlanMethodsCase):
    def test_default_mode_is_agent(self) -> None:
        sid = self._make_session()
        result = self._ok("plan.mode_get", {"session_id": sid})
        self.assertEqual(result, {"session_id": sid, "mode": "agent"})

    def test_returns_plan_after_set_mode(self) -> None:
        sid = self._make_session()
        set_mode(sid, SessionMode.PLAN)
        result = self._ok("plan.mode_get", {"session_id": sid})
        self.assertEqual(result["mode"], "plan")

    def test_missing_session_id_errors(self) -> None:
        err = self._err("plan.mode_get", {})
        self.assertEqual(err.code, ERROR_INVALID_PARAMS)
        self.assertIn("session_id", err.message)

    def test_blank_session_id_errors(self) -> None:
        err = self._err("plan.mode_get", {"session_id": "   "})
        self.assertEqual(err.code, ERROR_INVALID_PARAMS)

    def test_unknown_session_errors(self) -> None:
        err = self._err("plan.mode_get", {"session_id": "does-not-exist"})
        self.assertEqual(err.code, ERROR_APPLICATION)
        self.assertIn("unknown session", err.message)


# ---------------------------------------------------------------- mode_set

class ModeSetTests(_PlanMethodsCase):
    def test_set_plan_updates_session_state(self) -> None:
        sid = self._make_session()
        result = self._ok("plan.mode_set", {"session_id": sid, "mode": "plan"})
        self.assertEqual(result, {"session_id": sid, "mode": "plan"})
        # Subsequent get reflects the change.
        self.assertEqual(
            self._ok("plan.mode_get", {"session_id": sid})["mode"], "plan"
        )
        record = load_session(sid)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.mode, "plan")

    def test_set_agent_after_plan_round_trips(self) -> None:
        sid = self._make_session()
        self._ok("plan.mode_set", {"session_id": sid, "mode": "plan"})
        self._ok("plan.mode_set", {"session_id": sid, "mode": "agent"})
        self.assertEqual(
            self._ok("plan.mode_get", {"session_id": sid})["mode"], "agent"
        )

    def test_unknown_mode_rejected(self) -> None:
        sid = self._make_session()
        err = self._err(
            "plan.mode_set", {"session_id": sid, "mode": "supervisor"}
        )
        self.assertEqual(err.code, ERROR_INVALID_PARAMS)
        self.assertIn("unsupported plan mode", err.message)
        # Side-effect-free: state still agent.
        self.assertEqual(
            self._ok("plan.mode_get", {"session_id": sid})["mode"], "agent"
        )

    def test_non_string_mode_rejected(self) -> None:
        sid = self._make_session()
        err = self._err("plan.mode_set", {"session_id": sid, "mode": 42})
        self.assertEqual(err.code, ERROR_INVALID_PARAMS)

    def test_missing_mode_rejected(self) -> None:
        sid = self._make_session()
        err = self._err("plan.mode_set", {"session_id": sid})
        self.assertEqual(err.code, ERROR_INVALID_PARAMS)
        self.assertIn("mode", err.message)


# ---------------------------------------------------------------- current

class PlanCurrentTests(_PlanMethodsCase):
    def test_returns_mode_and_empty_artifact(self) -> None:
        sid = self._make_session()
        result = self._ok("plan.current", {"session_id": sid})
        self.assertEqual(result["session_id"], sid)
        self.assertEqual(result["mode"], "agent")
        self.assertFalse(result["has_plan"])
        self.assertIsNone(result["plan_content"])
        self.assertEqual(result["plan_path"], str(get_plan_path(sid)))

    def test_reflects_plan_mode_when_set(self) -> None:
        sid = self._make_session()
        self._ok("plan.mode_set", {"session_id": sid, "mode": "plan"})
        result = self._ok("plan.current", {"session_id": sid})
        self.assertEqual(result["mode"], "plan")

    def test_unknown_session_errors(self) -> None:
        err = self._err("plan.current", {"session_id": "missing"})
        self.assertEqual(err.code, ERROR_APPLICATION)


# ---------------------------------------------------------------- clear

class PlanClearTests(_PlanMethodsCase):
    def test_clears_artifact_and_resets_mode(self) -> None:
        sid = self._make_session()
        set_mode(sid, SessionMode.PLAN)
        write_plan(sid, "# Plan\n")

        result = self._ok("plan.clear", {"session_id": sid})

        self.assertEqual(result["session_id"], sid)
        self.assertEqual(result["mode"], "agent")
        self.assertEqual(result["plan_path"], str(get_plan_path(sid)))
        self.assertFalse(result["has_plan"])
        self.assertIsNone(read_plan(sid))
        self.assertEqual(self._ok("plan.mode_get", {"session_id": sid})["mode"], "agent")

    def test_clear_unknown_session_errors(self) -> None:
        err = self._err("plan.clear", {"session_id": "missing"})
        self.assertEqual(err.code, ERROR_APPLICATION)


# ---------------------------------------------------------------- session.current

class SessionCurrentModeFieldTests(_PlanMethodsCase):
    def test_session_current_includes_mode_field(self) -> None:
        sid = self._make_session()
        # ``session.resume`` is a long handler whose response goes out
        # via the transport; for this assertion all we need is the
        # current-session binding, which we set directly.
        set_current_session(sid)
        self.addCleanup(set_current_session, None)
        result = self._ok("session.current")
        self.assertIsNotNone(result["info"])
        # mode is additive — old clients tolerate its absence; new ones
        # see "agent" by default.
        self.assertEqual(result["info"].get("mode"), "agent")

    def test_session_current_mode_updates_after_set(self) -> None:
        sid = self._make_session()
        set_current_session(sid)
        self.addCleanup(set_current_session, None)
        self._ok("plan.mode_set", {"session_id": sid, "mode": "plan"})
        result = self._ok("session.current")
        self.assertEqual(result["info"]["mode"], "plan")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
