"""Tests for the ``session.*`` gateway methods (PR 3)."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from aether.cli.sessions import (
    SessionRecord,
    delete_session,
    load_session,
    save_session,
)
from aether.gateway.dispatcher import (
    _LONG_METHODS,
    dispatch_request,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.handlers.state import reset_state_for_tests
from aether.gateway.protocol import (
    ERROR_APPLICATION,
    ERROR_INVALID_PARAMS,
    RpcRequest,
    RpcResponse,
)
from aether.gateway.transport import (
    StdioTransport,
    bind_transport,
    reset_transport,
    reset_transport_for_tests,
)
from aether.runtime.session.plan_artifact import read_plan, write_plan
from aether.runtime.session.session_state import SessionMode, get_mode, set_mode


class _SessionMethodsCase(unittest.TestCase):
    """Each test isolates ``AETHER_HOME`` so session files don't leak."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = mock.patch.dict(
            os.environ, {"AETHER_HOME": self._tmp.name}
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        reset_dispatcher_for_tests()
        reset_transport_for_tests()
        reset_state_for_tests()
        register_builtins()
        register_handler_methods()

        # Bind a captured-stdout transport so long handlers (which
        # write their response asynchronously from the worker pool)
        # can be observed by ``_call`` below.
        self._buf = io.StringIO()
        self._sink = StdioTransport(lambda: self._buf)
        self._token = bind_transport(self._sink)
        self.addCleanup(reset_transport, self._token)

    def _call(self, name: str, params: dict | None = None) -> RpcResponse:
        # Reset buffer so each call inspects only its own response.
        self._buf.seek(0)
        self._buf.truncate(0)

        request = RpcRequest(id=name, method=name, params=params)
        resp = dispatch_request(request, transport=self._sink)
        if resp is not None:
            return resp

        # Long handler — response comes from the worker pool.
        if name not in _LONG_METHODS:
            self.fail(
                f"{name} returned None but is not marked long=True; "
                "the dispatcher contract was violated"
            )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._buf.getvalue().strip():
                break
            time.sleep(0.01)
        line = self._buf.getvalue().strip()
        if not line:
            self.fail(
                f"{name} (long) did not produce a response within 2s"
            )
        parsed = json.loads(line)
        return RpcResponse.model_validate(parsed)

    def _result(self, name: str, params: dict | None = None) -> dict:
        resp = self._call(name, params)
        if resp.error is not None:
            self.fail(
                f"{name} returned error: code={resp.error.code} "
                f"message={resp.error.message}"
            )
        assert resp.result is not None
        return resp.result


class CreateSession(_SessionMethodsCase):
    def test_creates_and_persists(self) -> None:
        result = self._result(
            "session.create",
            {"provider": "claude", "model": "claude-sonnet-4-6"},
        )
        session_id = result["session_id"]
        self.assertEqual(len(session_id), 36)  # uuid
        info = result["info"]
        self.assertEqual(info["session_id"], session_id)
        self.assertEqual(info["provider"], "claude")
        self.assertEqual(info["model"], "claude-sonnet-4-6")
        self.assertEqual(info["message_count"], 0)
        self.assertGreater(info["created_at"], 0)

        on_disk = load_session(session_id)
        self.assertIsNotNone(on_disk)
        assert on_disk is not None
        self.assertEqual(on_disk.provider, "claude")

    def test_requires_provider_and_model(self) -> None:
        for params in (None, {}, {"provider": "claude"}, {"model": "x"}):
            resp = self._call("session.create", params)
            assert resp.error is not None
            self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)

    def test_sets_current_session(self) -> None:
        result = self._result(
            "session.create",
            {"provider": "claude", "model": "claude-sonnet-4-6"},
        )
        current = self._result("session.current")
        self.assertEqual(current["session_id"], result["session_id"])

    def test_accepts_requested_session_id(self) -> None:
        result = self._result(
            "session.create",
            {
                "session_id": "ses_custom",
                "provider": "claude",
                "model": "claude-sonnet-4-6",
            },
        )
        self.assertEqual(result["session_id"], "ses_custom")
        self.assertIsNotNone(load_session("ses_custom"))

    def test_rejects_empty_requested_session_id(self) -> None:
        resp = self._call(
            "session.create",
            {
                "session_id": " ",
                "provider": "claude",
                "model": "claude-sonnet-4-6",
            },
        )
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)

    def test_optional_system_prompt(self) -> None:
        result = self._result(
            "session.create",
            {
                "provider": "claude",
                "model": "claude-sonnet-4-6",
                "system": "you are helpful",
            },
        )
        self.assertEqual(result["info"]["system_prompt"], "you are helpful")


class ListSessions(_SessionMethodsCase):
    def test_empty_initial_state(self) -> None:
        result = self._result("session.list")
        self.assertEqual(result["sessions"], [])

    def test_lists_persisted_sessions(self) -> None:
        for i in range(3):
            self._result(
                "session.create",
                {"provider": "claude", "model": f"claude-test-{i}"},
            )
        result = self._result("session.list")
        self.assertEqual(len(result["sessions"]), 3)

    def test_limit_param_respected(self) -> None:
        for i in range(5):
            self._result(
                "session.create",
                {"provider": "claude", "model": f"claude-test-{i}"},
            )
        result = self._result("session.list", {"limit": 2})
        self.assertEqual(len(result["sessions"]), 2)


class ResumeSession(_SessionMethodsCase):
    def test_returns_info_and_messages(self) -> None:
        record = SessionRecord.new(
            session_id="ses_resume_1",
            provider="claude",
            model="claude-sonnet-4-6",
        )
        record.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        record.first_user_message = "hello"
        save_session(record)

        result = self._result("session.resume", {"session_id": "ses_resume_1"})
        self.assertEqual(result["info"]["session_id"], "ses_resume_1")
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][0]["text"], "hello")
        self.assertEqual(result["messages"][1]["role"], "assistant")

    def test_unknown_session_raises_application_error(self) -> None:
        resp = self._call("session.resume", {"session_id": "does-not-exist"})
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_APPLICATION)
        self.assertIn("does-not-exist", resp.error.message)

    def test_updates_current_session(self) -> None:
        record = SessionRecord.new(
            session_id="ses_current_via_resume",
            provider="claude",
            model="claude-sonnet-4-6",
        )
        save_session(record)
        self._result("session.resume", {"session_id": "ses_current_via_resume"})
        current = self._result("session.current")
        self.assertEqual(current["session_id"], "ses_current_via_resume")

    def test_resume_restores_persisted_plan_mode(self) -> None:
        record = SessionRecord.new(
            session_id="ses_plan_resume",
            provider="claude",
            model="claude-sonnet-4-6",
        )
        record.mode = "plan"
        save_session(record)

        result = self._result("session.resume", {"session_id": "ses_plan_resume"})

        self.assertEqual(result["info"]["mode"], "plan")
        self.assertEqual(get_mode("ses_plan_resume"), "plan")

    def test_accepts_unique_prefix(self) -> None:
        record = SessionRecord.new(
            session_id="ses_prefix_target",
            provider="claude",
            model="claude-sonnet-4-6",
        )
        save_session(record)
        result = self._result("session.resume", {"session_id": "ses_prefix"})
        self.assertEqual(result["info"]["session_id"], "ses_prefix_target")

    def test_ambiguous_prefix_raises_application_error(self) -> None:
        for session_id in ("ses_same_1", "ses_same_2"):
            save_session(
                SessionRecord.new(
                    session_id=session_id,
                    provider="claude",
                    model="claude-sonnet-4-6",
                )
            )
        resp = self._call("session.resume", {"session_id": "ses_same"})
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_APPLICATION)
        self.assertIn("ambiguous", resp.error.message)

    def test_tolerates_unknown_roles(self) -> None:
        record = SessionRecord.new(
            session_id="ses_weird_role",
            provider="claude",
            model="claude-sonnet-4-6",
        )
        record.messages = [{"role": "developer", "content": "system-ish"}]
        save_session(record)
        result = self._result("session.resume", {"session_id": "ses_weird_role"})
        # Unknown role coerced to ``user`` per the schema-tolerance contract.
        self.assertEqual(result["messages"][0]["role"], "user")

    def test_assistant_tool_calls_normalised_into_wire_shape(self) -> None:
        record = SessionRecord.new(
            session_id="ses_tool_calls",
            provider="openai",
            model="gpt-x",
        )
        record.messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tool_a",
                        "type": "function",
                        "function": {
                            "name": "list_dir",
                            "arguments": {"path": "/x"},
                        },
                    },
                    {
                        "id": "tool_b",
                        "type": "function",
                        # Providers sometimes ship arguments as a JSON
                        # string — round-trip must parse it.
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/y"}',
                        },
                    },
                ],
            },
        ]
        save_session(record)
        result = self._result(
            "session.resume", {"session_id": "ses_tool_calls"}
        )
        assistant = result["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        tool_calls = assistant.get("tool_calls")
        assert isinstance(tool_calls, list)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["id"], "tool_a")
        self.assertEqual(tool_calls[0]["name"], "list_dir")
        self.assertEqual(tool_calls[0]["arguments"], {"path": "/x"})
        self.assertEqual(tool_calls[1]["id"], "tool_b")
        self.assertEqual(tool_calls[1]["name"], "read_file")
        self.assertEqual(tool_calls[1]["arguments"], {"path": "/y"})

    def test_tool_role_forwards_is_error_and_metadata(self) -> None:
        record = SessionRecord.new(
            session_id="ses_tool_result",
            provider="openai",
            model="gpt-x",
        )
        record.messages = [
            {
                "role": "tool",
                "tool_call_id": "tool_a",
                "name": "file_edit",
                "content": "edited /x",
                "is_error": False,
                "metadata": {
                    "path": "/x",
                    "lines_added": 5,
                    "lines_removed": 2,
                },
            },
            {
                "role": "tool",
                "tool_call_id": "tool_b",
                "name": "shell",
                "content": "boom",
                "is_error": True,
                "metadata": {"exit_code": 2},
            },
        ]
        save_session(record)
        result = self._result(
            "session.resume", {"session_id": "ses_tool_result"}
        )
        ok = result["messages"][0]
        self.assertEqual(ok["role"], "tool")
        self.assertEqual(ok["tool_call_id"], "tool_a")
        self.assertEqual(ok["is_error"], False)
        self.assertEqual(ok["metadata"]["lines_added"], 5)
        bad = result["messages"][1]
        self.assertEqual(bad["is_error"], True)
        self.assertEqual(bad["metadata"]["exit_code"], 2)

    def test_malformed_tool_call_arguments_round_trip(self) -> None:
        record = SessionRecord.new(
            session_id="ses_bad_args",
            provider="openai",
            model="gpt-x",
        )
        record.messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tool_bad",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            # Malformed JSON — must not crash; surfaced
                            # to the TUI under __raw__.
                            "arguments": "{not json",
                        },
                    }
                ],
            }
        ]
        save_session(record)
        result = self._result(
            "session.resume", {"session_id": "ses_bad_args"}
        )
        tool_calls = result["messages"][0]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["arguments"], {"__raw__": "{not json"})


class UpdateSession(_SessionMethodsCase):
    def test_updates_model_without_clearing_messages(self) -> None:
        record = SessionRecord.new(
            session_id="ses_model_switch",
            provider="openai",
            model="old-model",
        )
        record.messages = [{"role": "user", "content": "keep this"}]
        save_session(record)

        result = self._result(
            "session.update",
            {
                "session_id": "ses_model_switch",
                "provider": "openai",
                "model": "new-model",
            },
        )
        self.assertEqual(result["info"]["model"], "new-model")
        on_disk = load_session("ses_model_switch")
        self.assertIsNotNone(on_disk)
        assert on_disk is not None
        self.assertEqual(on_disk.model, "new-model")
        self.assertEqual(on_disk.messages[0]["content"], "keep this")

    def test_unknown_session_raises_application_error(self) -> None:
        resp = self._call("session.update", {"session_id": "missing", "model": "x"})
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_APPLICATION)

    def test_rejects_empty_model(self) -> None:
        record = SessionRecord.new(
            session_id="ses_bad_update",
            provider="openai",
            model="old-model",
        )
        save_session(record)
        resp = self._call(
            "session.update",
            {"session_id": "ses_bad_update", "model": " "},
        )
        assert resp.error is not None
        self.assertEqual(resp.error.code, ERROR_INVALID_PARAMS)


class DeleteSession(_SessionMethodsCase):
    def test_deletes_existing(self) -> None:
        record = SessionRecord.new(
            session_id="ses_to_delete",
            provider="claude",
            model="x",
        )
        save_session(record)
        result = self._result("session.delete", {"session_id": "ses_to_delete"})
        self.assertTrue(result["deleted"])
        self.assertIsNone(load_session("ses_to_delete"))

    def test_unknown_returns_false_not_error(self) -> None:
        result = self._result("session.delete", {"session_id": "phantom"})
        self.assertFalse(result["deleted"])

    def test_deleting_current_clears_current(self) -> None:
        created = self._result(
            "session.create",
            {"provider": "claude", "model": "claude-sonnet-4-6"},
        )
        sid = created["session_id"]
        self._result("session.delete", {"session_id": sid})
        current = self._result("session.current")
        self.assertIsNone(current["session_id"])


class CurrentSession(_SessionMethodsCase):
    def test_null_when_nothing_bound(self) -> None:
        result = self._result("session.current")
        self.assertIsNone(result["session_id"])
        self.assertNotIn("info", result)

    def test_returns_info_when_bound(self) -> None:
        created = self._result(
            "session.create",
            {"provider": "claude", "model": "claude-sonnet-4-6"},
        )
        current = self._result("session.current")
        self.assertEqual(current["session_id"], created["session_id"])
        self.assertEqual(current["info"]["provider"], "claude")

    def test_create_resets_reused_session_plan_state(self) -> None:
        sid = "ses_reused_plan"
        set_mode(sid, SessionMode.PLAN)
        write_plan(sid, "old plan")

        created = self._result(
            "session.create",
            {
                "session_id": sid,
                "provider": "claude",
                "model": "claude-sonnet-4-6",
            },
        )

        self.assertEqual(created["session_id"], sid)
        self.assertEqual(get_mode(sid), "agent")
        self.assertIsNone(read_plan(sid))

    def test_clears_stale_pointer(self) -> None:
        """If the on-disk session is deleted out of band, current must null out."""
        created = self._result(
            "session.create",
            {"provider": "claude", "model": "claude-sonnet-4-6"},
        )
        sid = created["session_id"]
        # Delete the file directly — bypasses our delete handler.
        delete_session(sid)
        result = self._result("session.current")
        self.assertIsNone(result["session_id"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
