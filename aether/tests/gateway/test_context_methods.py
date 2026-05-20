from __future__ import annotations

import io
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from aether.cli.sessions import SessionRecord, load_session, save_session
from aether.gateway.dispatcher import (
    _LONG_METHODS,
    dispatch_request,
    register_builtins,
    reset_dispatcher_for_tests,
)
from aether.gateway.handlers import register_handler_methods
from aether.gateway.handlers import context_methods
from aether.gateway.handlers.state import reset_state_for_tests, set_current_session
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
from aether.runtime.context import CompressionResult


class _FakeCompressionService:
    def __init__(self, result: CompressionResult) -> None:
        self.result = result
        self.requests = []

    def compress(self, request):
        self.requests.append(request)
        return self.result


class ContextMethodsCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = mock.patch.dict(os.environ, {"AETHER_HOME": self._tmp.name})
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        reset_dispatcher_for_tests()
        reset_transport_for_tests()
        reset_state_for_tests()
        context_methods.reset_context_status_for_tests()
        register_builtins()
        register_handler_methods()

        self._buf = io.StringIO()
        self._sink = StdioTransport(lambda: self._buf)
        self._token = bind_transport(self._sink)
        self.addCleanup(reset_transport, self._token)

    def _save_session(
        self,
        *,
        session_id: str = "ctx-s",
        messages: list[dict] | None = None,
        provider: str = "openai",
        model: str = "gpt-5.4",
    ) -> SessionRecord:
        record = SessionRecord.new(
            session_id=session_id,
            provider=provider,
            model=model,
        )
        record.messages = messages if messages is not None else []
        save_session(record)
        return record

    def _call(self, name: str, params: dict | None = None) -> RpcResponse:
        self._buf.seek(0)
        self._buf.truncate(0)
        request = RpcRequest(id=name, method=name, params=params)
        resp = dispatch_request(request, transport=self._sink)
        if resp is not None:
            return resp
        if name not in _LONG_METHODS:
            self.fail(f"{name} returned async response but is not long=True")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._buf.getvalue().strip():
                break
            time.sleep(0.01)
        line = self._buf.getvalue().strip()
        if not line:
            self.fail(f"{name} did not respond")
        return RpcResponse.model_validate(json.loads(line))

    def _result(self, name: str, params: dict | None = None) -> dict:
        resp = self._call(name, params)
        if resp.error is not None:
            self.fail(f"{name} returned error: {resp.error.code} {resp.error.message}")
        assert resp.result is not None
        return resp.result

    def _error(self, name: str, params: dict | None = None):
        resp = self._call(name, params)
        if resp.error is None:
            self.fail(f"{name} expected error, got {resp.result!r}")
        return resp.error


class ContextStatusTests(ContextMethodsCase):
    def test_missing_session_errors(self) -> None:
        err = self._error("context.status", {"session_id": "missing"})
        self.assertEqual(err.code, ERROR_APPLICATION)

    def test_status_defaults_for_existing_session(self) -> None:
        self._save_session(messages=[{"role": "user", "content": "hello"}])

        result = self._result("context.status", {"session_id": "ctx-s"})

        self.assertEqual(result["session_id"], "ctx-s")
        self.assertEqual(result["context_engine"], "default")
        self.assertEqual(result["compression_count"], 0)
        self.assertIsNone(result["last_compression"])
        self.assertEqual(result["message_count"], 1)


class ContextCompressTests(ContextMethodsCase):
    def test_missing_session_id_errors_without_current_session(self) -> None:
        err = self._error("context.compress", {})
        self.assertEqual(err.code, ERROR_INVALID_PARAMS)

    def test_skips_when_not_enough_context(self) -> None:
        self._save_session(messages=[{"role": "user", "content": "hello"}])

        result = self._result("context.compress", {"session_id": "ctx-s"})

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["last_compression"]["reason"], "not_enough_context")
        self.assertEqual(result["compression_count"], 0)

    def test_success_updates_session_messages_and_status(self) -> None:
        messages = [
            {"role": "user", "content": f"message {idx}"}
            for idx in range(6)
        ]
        self._save_session(messages=messages)
        fake = _FakeCompressionService(
            CompressionResult(
                messages=[{"role": "user", "content": "summary"}],
                status="compressed",
                metadata={
                    "status": "compressed",
                    "trigger_reason": "manual",
                    "source_message_count": 6,
                    "result_message_count": 1,
                },
            )
        )

        with mock.patch.object(context_methods, "_build_compression_service", return_value=fake):
            result = self._result(
                "context.compress",
                {"session_id": "ctx-s", "focus": "auth", "force": True},
            )

        self.assertEqual(result["status"], "compressed")
        self.assertEqual(result["compression_count"], 1)
        self.assertEqual(result["last_compression"]["source_message_count"], 6)
        self.assertEqual(fake.requests[0].focus, "auth")
        saved = load_session("ctx-s")
        assert saved is not None
        self.assertEqual(saved.messages, [{"role": "user", "content": "summary"}])

        status = self._result("context.status", {"session_id": "ctx-s"})
        self.assertEqual(status["compression_count"], 1)

    def test_failure_preserves_session_messages(self) -> None:
        messages = [
            {"role": "user", "content": f"message {idx}"}
            for idx in range(6)
        ]
        self._save_session(messages=messages)
        fake = _FakeCompressionService(
            CompressionResult(
                messages=messages,
                status="failed",
                metadata={"status": "failed", "error": "RuntimeError"},
                error="RuntimeError",
            )
        )

        with mock.patch.object(context_methods, "_build_compression_service", return_value=fake):
            result = self._result("context.compress", {"session_id": "ctx-s"})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "RuntimeError")
        saved = load_session("ctx-s")
        assert saved is not None
        self.assertEqual(saved.messages, messages)

    def test_uses_current_session_when_session_id_omitted(self) -> None:
        self._save_session(messages=[{"role": "user", "content": "hello"}])
        set_current_session("ctx-s")

        result = self._result("context.compress", {})

        self.assertEqual(result["session_id"], "ctx-s")
        self.assertEqual(result["status"], "skipped")
